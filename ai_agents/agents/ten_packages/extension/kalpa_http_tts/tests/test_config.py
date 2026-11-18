#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from kalpa_http_tts.config import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    SAMPLE_RATES,
    KalpaHttpTTSConfig,
)
from kalpa_http_tts.extension import (
    KalpaHttpTTSExtension,
)
from kalpa_http_tts.kalpa_tts import (
    _error_from_response,
    _error_from_transport,
    _split,
)

KEY = "sk-live-0123456789abcdefghij"
PACKAGE = Path(__file__).parents[1]


def load(properties: dict) -> KalpaHttpTTSConfig:
    config = KalpaHttpTTSConfig(**properties)
    config.update_params()
    return config


def refusal(status: int, body: bytes, **headers: str):
    response = httpx.Response(
        status,
        content=body,
        headers={
            name.replace("_", "-"): value for name, value in headers.items()
        },
    )
    return asyncio.run(_error_from_response(response))


def test_package_import_registers_the_addon():
    import kalpa_http_tts

    # Importing the package is what runs the registration decorator.
    assert kalpa_http_tts.addon.KalpaHttpTTSExtensionAddon.__name__ == (
        "KalpaHttpTTSExtensionAddon"
    )


def test_manifest_properties_and_versions_agree():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    declared = manifest["api"]["property"]["properties"]
    assert manifest["name"] == PACKAGE.name == "kalpa_http_tts"
    assert {"url", "base_url"} <= set(declared["params"]["properties"])
    assert "url" in declared
    assert (
        f'version = "{manifest["version"]}"'
        in (PACKAGE / "pyproject.toml").read_text()
    )
    # Every setting of the defaults and of the guarder configs is declared.
    for path in [PACKAGE / "property.json", *PACKAGE.glob("tests/configs/*")]:
        properties = json.loads(path.read_text())
        assert set(properties) <= set(declared), path.name
        assert set(properties["params"]) <= set(
            declared["params"]["properties"]
        ), path.name
    # No key is written down anywhere.
    assert json.loads((PACKAGE / "property.json").read_text())["params"][
        "api_key"
    ].startswith("${env:")


def test_defaults_need_only_a_key():
    config = load({"params": {"api_key": KEY}})
    config.validate()
    assert config.model == DEFAULT_MODEL
    assert config.voice_id == DEFAULT_VOICE_ID
    assert config.sample_rate == DEFAULT_SAMPLE_RATE == 24000
    assert config.dump is False
    assert (
        config.endpoint
        == f"https://api.kalpalabs.ai/v1/tts/{DEFAULT_VOICE_ID}/stream"
    )
    # The key is not part of the body; it rides the Authorization header.
    assert config.request_body("Hello.") == {
        "text": "Hello.",
        "model": DEFAULT_MODEL,
        "output_format": "pcm_s16le",
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }


def test_missing_key_or_voice_is_rejected():
    with pytest.raises(ValueError, match="api_key"):
        load({"params": {"api_key": ""}}).validate()
    with pytest.raises(ValueError, match="api_key"):
        load({"params": {"api_key": None}}).validate()
    with pytest.raises(ValueError, match="voice_id"):
        load({"params": {"api_key": KEY}, "voice_id": ""}).validate()


def test_settings_leave_params_and_the_rest_is_passed_through():
    config = load(
        {
            "params": {
                "api_key": KEY,
                "voice_id": "voice-1",
                "model": "kalpa-tts-2",
                "sample_rate": "16000",
                "temperature": 0.7,
                "audio_quality": "medium",
                # The extension sets these itself.
                "output_format": "mp3",
                "text": "not this",
                # Settings of the WebSocket endpoint only.
                "chunk_length_schedule": [50, 120],
                "generation_config": {"chunk_length_schedule": [50]},
            }
        }
    )
    config.validate()
    assert config.api_key == KEY
    assert config.endpoint == "https://api.kalpalabs.ai/v1/tts/voice-1/stream"
    assert config.request_body("Hi.") == {
        "text": "Hi.",
        "model": "kalpa-tts-2",
        "output_format": "pcm_s16le",
        "sample_rate": 16000,
        "params": {"temperature": 0.7, "audio_quality": "medium"},
    }
    # A request does not change the parameters of the next one.
    config.request_body("Hi.")["params"]["temperature"] = 1.5
    assert config.params == {"temperature": 0.7, "audio_quality": "medium"}


@pytest.mark.parametrize("rate", SAMPLE_RATES)
def test_every_rate_kalpa_offers_is_accepted(rate):
    config = load({"params": {"api_key": KEY, "sample_rate": rate}})
    config.validate()
    assert config.sample_rate == rate
    assert config.request_body("Hi.")["sample_rate"] == rate


@pytest.mark.parametrize(
    "rate",
    [
        0,
        11025,
        96000,
        -24000,
        24000.5,
        True,
        "fast",
        "inf",
        "nan",
        None,
        [8000],
    ],
)
def test_any_other_rate_is_rejected(rate):
    config = load({"params": {"api_key": KEY, "sample_rate": rate}})
    with pytest.raises(ValueError, match="sample_rate"):
        config.validate()


def test_endpoint_precedence():
    def endpoint(properties: dict) -> str:
        properties.setdefault("params", {})["api_key"] = KEY
        config = load(properties)
        config.validate()
        # None of them is sent to Kalpa as a generation parameter.
        assert "params" not in config.request_body("Hi.")
        return config.endpoint

    # params.url first, then the top-level url, then params.base_url.
    assert (
        endpoint(
            {
                "url": "https://top.test/speak",
                "params": {
                    "url": "https://nested.test/{voice_id}",
                    "base_url": "https://base.test",
                    "voice_id": "v1",
                },
            }
        )
        == "https://nested.test/v1"
    )
    assert (
        endpoint(
            {
                "url": "https://top.test/{voice_id}/speak",
                "params": {"base_url": "https://base.test", "voice_id": "v1"},
            }
        )
        == "https://top.test/v1/speak"
    )
    assert (
        endpoint(
            {"params": {"base_url": "http://base.test/", "voice_id": "v1"}}
        )
        == "http://base.test/v1/tts/v1/stream"
    )
    # A params.url that is blank or not a string does not count.
    for nothing in ("", "   ", None, 7, ["https://list.test"]):
        assert (
            endpoint(
                {"url": "https://top.test/speak", "params": {"url": nothing}}
            )
            == "https://top.test/speak"
        )


def test_the_endpoint_is_reported_without_what_may_be_a_secret():
    def safe(url: str) -> str:
        return load({"params": {"api_key": KEY, "url": url}}).safe_url

    assert (
        safe("https://user:pass@kalpa.test:8443/v1/{voice_id}?token=t#frag")
        == f"https://kalpa.test:8443/v1/{DEFAULT_VOICE_ID}"
    )
    assert safe("http://[::1]:8080/speak?x=1") == "http://[::1]:8080/speak"
    assert safe("https://kalpa.test/speak") == "https://kalpa.test/speak"
    # A URL that cannot be read is not reported at all.
    assert safe("https://kalpa.test:port/speak") == ""
    assert (
        load({"params": {"api_key": KEY}}).safe_url
        == f"https://api.kalpalabs.ai/v1/tts/{DEFAULT_VOICE_ID}/stream"
    )


def test_an_endpoint_that_is_not_http_is_rejected():
    config = load({"params": {"api_key": KEY, "base_url": "wss://base.test"}})
    with pytest.raises(ValueError, match="http"):
        config.validate()


def test_neither_the_key_nor_a_secret_in_a_url_is_printed():
    config = load(
        {
            "url": f"https://top.test/speak?access={KEY}",
            "params": {
                "api_key": KEY,
                "base_url": f"https://user:{KEY}@base.test?api_key={KEY}",
                "temperature": 0.7,
            },
        }
    )
    printed = config.to_str()
    assert KEY not in printed
    assert "https://top.test/speak" in printed
    assert "https://base.test" in printed
    assert "temperature" in printed
    assert KEY in config.to_str(sensitive_handling=False)


@pytest.mark.parametrize(
    "key",
    [KEY + "\n", KEY + " ", "\t" + KEY, "sk live", KEY + "\x00", "cl\u00e9"],
)
def test_a_key_that_a_header_cannot_hold_is_rejected_unquoted(key):
    with pytest.raises(ValueError) as raised:
        load({"params": {"api_key": key}}).validate()
    assert str(raised.value) == (
        "'api_key' must not contain whitespace, control or non-ASCII "
        "characters."
    )


def test_unreadable_settings_are_reported_without_their_values():
    extension = KalpaHttpTTSExtension("kalpa")
    with pytest.raises(ValueError) as raised:
        # The key has ended up where a table of settings belongs.
        asyncio.run(extension.create_config(json.dumps({"params": KEY})))
    assert "params" in str(raised.value)
    assert KEY not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_a_long_text_is_cut_at_a_space():
    assert _split("short", 10) == ["short"]
    assert _split("one two three four", 10) == ["one two ", "three four"]
    assert _split("abcdefghijklmno", 10) == ["abcdefghij", "klmno"]
    pieces = _split("word " * 2000, 8000)
    assert "".join(pieces) == "word " * 2000
    assert all(len(piece) <= 8000 for piece in pieces)


@pytest.mark.parametrize(
    ("status", "error_type", "permanent"),
    [
        (401, "authentication_error", True),
        (402, "allowance_exhausted", True),
        (403, "account_suspended", True),
        (404, "not_found", True),
        (422, "invalid_request", True),
        (400, "context_length_exceeded", False),
        (429, "rate_limit_exceeded", False),
        (500, "internal_error", False),
        (502, "inference_error", False),
        (503, "overloaded", False),
        (503, "service_unavailable", False),
    ],
)
def test_kalpa_errors_keep_their_type(status, error_type, permanent):
    body = {"error": {"type": error_type, "message": "Why.", "request_id": "r"}}
    error = refusal(status, json.dumps(body).encode())
    assert (error.error_type, error.message) == (error_type, "Why.")
    assert error.permanent is permanent
    assert (error.status, error.request_id) == (status, "r")
    assert str(error) == f"{error_type}: Why."


def test_an_answer_without_kalpas_error_body_is_named_by_its_status():
    # A proxy in front of Kalpa answers in its own words. Its 403 is not
    # Kalpa's `account_suspended`.
    error = refusal(403, b"<html>Forbidden</html>", x_request_id="abc")
    assert (error.error_type, error.permanent) == ("http_403", True)
    assert error.message == "Kalpa answered with HTTP 403"
    assert error.request_id == "abc"
    for status in (401, 402, 404):
        error = refusal(status, b"")
        assert (error.error_type, error.permanent) == (f"http_{status}", True)
    for status in (400, 429, 500, 502, 503, 504):
        error = refusal(status, b"gateway says no")
        assert (error.error_type, error.permanent) == (
            f"http_{status}",
            False,
        )
    error = refusal(500, b'["not", "an", "object"]')
    assert (error.error_type, error.permanent) == ("http_500", False)
    error = refusal(403, b'{"error": "a string"}')
    assert (error.error_type, error.permanent) == ("http_403", True)
    # With Kalpa's body, the type decides, whatever the status.
    body = b'{"error": {"type": "inference_error", "message": "Why."}}'
    assert refusal(403, body).permanent is False


def test_a_failure_of_the_connection_is_reported_but_not_of_the_request():
    lost = _error_from_transport(httpx.ReadError("connection reset"))
    assert str(lost) == "connection_error: connection reset"
    lost = _error_from_transport(httpx.RemoteProtocolError(""))
    assert str(lost) == "connection_error: RemoteProtocolError"
    late = _error_from_transport(httpx.ConnectTimeout("too slow"))
    assert str(late) == "timeout: Kalpa did not answer in time"
    # What the HTTP library says about a request it will not send can
    # quote a header.
    for unsent in (
        httpx.LocalProtocolError(f"Illegal header value b'Bearer {KEY}'"),
        httpx.UnsupportedProtocol(f"no adapter for {KEY}"),
    ):
        error = _error_from_transport(unsent)
        assert error.error_type == "request_error"
        assert KEY not in str(error)
        assert type(unsent).__name__ in error.message
        assert not error.permanent


@pytest.mark.parametrize(
    ("header", "seconds"),
    [("1", 1.0), ("2.5", 2.5), ("", 0.0), ("soon", 0.0), ("-1", 0.0)],
)
def test_retry_after_is_read_in_seconds(header, seconds):
    body = b'{"error": {"type": "overloaded", "message": "Busy."}}'
    assert refusal(503, body, retry_after=header).retry_after == seconds
