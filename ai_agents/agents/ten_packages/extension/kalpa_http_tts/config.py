#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import copy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field
from ten_ai_base import utils
from ten_ai_base.tts2_http import AsyncTTS2HttpConfig

DEFAULT_BASE_URL = "https://api.kalpalabs.ai"
ENDPOINT_PATH = "/v1/tts/{voice_id}/stream"
DEFAULT_MODEL = "kalpa-tts-1"
DEFAULT_VOICE_ID = "18c26f49-9f9c-400c-bec9-7467f28e1db5"  # Arjun
DEFAULT_SAMPLE_RATE = 24000
# Kalpa generates at 24 kHz and resamples to the others on its side.
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# Keys of the request body that the extension sets itself, and settings of
# Kalpa's WebSocket endpoint that this one does not have. A value in
# `params` must not reach the generation parameters under one of these names.
_RESERVED_PARAMS = (
    "output_format",
    "text",
    "chunk_length_schedule",
    "generation_config",
)


def _whole_number(value: Any) -> int:
    """`value` as an int; 0, which is no sample rate, when it is not one.

    `true`, `24000.5` and `"fast"` are not one.
    """
    if isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if number.is_integer() else 0


def _without_credentials(url: str) -> str:
    """The scheme, host and path of a URL.

    User info and query parameters may hold a key or a token, so a URL is
    logged and reported without them.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
    except ValueError:
        return ""
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class KalpaHttpTTSConfig(AsyncTTS2HttpConfig):
    api_key: str = ""
    # Full HTTP endpoint. `{voice_id}` in it is replaced by the voice.
    # `params.url` replaces a value given here.
    url: str = ""
    base_url: str = ""
    voice_id: str = DEFAULT_VOICE_ID
    model: str = DEFAULT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    dump: bool = False
    dump_path: str = "/tmp"
    # After update_params(): Kalpa generation parameters only.
    params: dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        """Lift the settings the extension consumes out of `params`.

        What stays in `params` is passed to Kalpa as generation parameters
        (`temperature`, `acoustic_temperature`, `audio_quality`, ...).
        Nothing is rejected here; `validate()` does that.
        """
        params = self.params

        if "api_key" in params:
            self.api_key = str(params.pop("api_key") or "")

        # Endpoint precedence: params.url, the top-level url,
        # params.base_url, then the default. None of them is a generation
        # parameter.
        nested_url = params.pop("url", None)
        if isinstance(nested_url, str) and nested_url.strip():
            self.url = nested_url
        nested_base_url = params.pop("base_url", None)
        if nested_base_url:
            self.base_url = str(nested_base_url)

        for name in ("voice_id", "model"):
            value = params.pop(name, None)
            if value:
                setattr(self, name, str(value))

        if "sample_rate" in params:
            self.sample_rate = _whole_number(params.pop("sample_rate"))

        for name in _RESERVED_PARAMS:
            params.pop(name, None)

    def validate(self) -> None:
        """Raise ValueError naming the first setting that cannot be used."""
        if not self.api_key:
            raise ValueError("Required parameter 'api_key' is missing.")
        # The key is sent as a header. One with a space, a line break or
        # another character a header cannot hold is rejected here, without
        # quoting it: the HTTP library would quote it in its own error.
        if not all(
            char.isascii() and char.isprintable() and char != " "
            for char in self.api_key
        ):
            raise ValueError(
                "'api_key' must not contain whitespace, control or "
                "non-ASCII characters."
            )
        if not self.voice_id:
            raise ValueError("Required parameter 'voice_id' is missing.")
        # A boolean, a fraction and a word arrive here as 0.
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(
                f"'sample_rate' must be one of "
                f"{', '.join(str(rate) for rate in SAMPLE_RATES)}."
            )
        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError("The Kalpa endpoint must be an http(s) URL.")

    @property
    def endpoint(self) -> str:
        if self.url:
            return self.url.replace("{voice_id}", self.voice_id)
        base_url = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        return base_url + ENDPOINT_PATH.format(voice_id=self.voice_id)

    @property
    def safe_url(self) -> str:
        """The endpoint as it may be logged and reported."""
        return _without_credentials(self.endpoint)

    def request_body(self, text: str) -> dict[str, Any]:
        """The JSON body that asks Kalpa to speak `text`."""
        body: dict[str, Any] = {
            "text": text,
            "model": self.model,
            "output_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }
        if self.params:
            body["params"] = dict(self.params)
        return body

    def to_str(self, sensitive_handling: bool = True) -> str:
        if not sensitive_handling:
            return f"{self}"

        config = copy.deepcopy(self)
        if config.api_key:
            config.api_key = utils.encrypt(config.api_key)
        config.url = _without_credentials(config.url)
        config.base_url = _without_credentials(config.base_url)
        config.params = utils.redact_json(config.params)
        return f"{config}"
