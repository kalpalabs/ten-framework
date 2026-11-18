#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import copy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field
from ten_ai_base import utils

DEFAULT_BASE_URL = "wss://api.kalpalabs.ai"
ENDPOINT_PATH = "/v1/tts/{voice_id}/stream"
DEFAULT_MODEL = "kalpa-tts-1"
DEFAULT_VOICE_ID = "18c26f49-9f9c-400c-bec9-7467f28e1db5"  # Arjun
DEFAULT_SAMPLE_RATE = 24000
# The output rates Kalpa offers.
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# Keys of the Kalpa wire protocol that the extension sets itself. A value in
# `params` must not reach the generation parameters under one of these names.
_RESERVED_PARAMS = (
    "output_format",
    "generation_config",
    "type",
    "text",
    "flush",
)


def _integer(name: str, value: Any) -> int:
    """`value` as an integer. `true`, `24000.5` and `"fast"` are not one."""
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"'{name}' must be an integer.")


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


class KalpaTTSConfig(BaseModel):
    api_key: str = ""
    # Full WebSocket endpoint. `{voice_id}` in it is replaced by the voice.
    url: str = ""
    base_url: str = ""
    voice_id: str = DEFAULT_VOICE_ID
    model: str = DEFAULT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_length_schedule: list[int] | None = None
    dump: bool = False
    dump_path: str = "/tmp"
    # After update_params(): Kalpa generation parameters only.
    params: dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        """Lift the settings the extension consumes out of `params`.

        What stays in `params` is passed to Kalpa as generation parameters
        (`temperature`, `acoustic_temperature`, `audio_quality`, ...).
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
            self.sample_rate = _integer(
                "sample_rate", params.pop("sample_rate")
            )

        # Left out or empty, Kalpa's own schedule applies. Kalpa checks the
        # values of one that is given.
        schedule = params.pop("chunk_length_schedule", None)
        if schedule:
            if not isinstance(schedule, list):
                raise ValueError(
                    "'chunk_length_schedule' must be a list of integers."
                )
            self.chunk_length_schedule = [
                _integer("chunk_length_schedule", item) for item in schedule
            ]

        for name in _RESERVED_PARAMS:
            params.pop(name, None)

    def check(self) -> None:
        """Raise ValueError naming the first setting that cannot be used."""
        if not self.api_key:
            raise ValueError("Required parameter 'api_key' is missing.")
        if not self.voice_id:
            raise ValueError("Required parameter 'voice_id' is missing.")
        # A boolean is an integer to Python: `True` would be sent as 1.
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(
                f"'sample_rate' must be one of "
                f"{', '.join(str(rate) for rate in SAMPLE_RATES)}."
            )

    @property
    def websocket_url(self) -> str:
        if self.url:
            return self.url.replace("{voice_id}", self.voice_id)
        base_url = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        return base_url + ENDPOINT_PATH.format(voice_id=self.voice_id)

    @property
    def safe_url(self) -> str:
        """The endpoint as it may be logged and reported."""
        return _without_credentials(self.websocket_url)

    def session_message(self) -> dict[str, Any]:
        """The `initializeConnection` message. It carries the API key."""
        generation_config: dict[str, Any] = {}
        if self.chunk_length_schedule:
            generation_config["chunk_length_schedule"] = (
                self.chunk_length_schedule
            )
        message: dict[str, Any] = {
            "type": "initializeConnection",
            "api_key": self.api_key,
            "model": self.model,
            "output_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            # Present, even when empty: audio starts at the first complete
            # sentences instead of waiting for the end of the request.
            "generation_config": generation_config,
        }
        if self.params:
            message["params"] = dict(self.params)
        return message

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
