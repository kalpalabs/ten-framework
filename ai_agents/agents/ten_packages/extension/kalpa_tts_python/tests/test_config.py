#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from ten_packages.extension.kalpa_tts_python.config import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    SAMPLE_RATES,
    KalpaTTSConfig,
)
from ten_packages.extension.kalpa_tts_python.extension import _split
from ten_packages.extension.kalpa_tts_python.kalpa_tts import (
    _error_from_close,
    _error_from_message,
)

KEY = "sk-live-0123456789abcdefghij"


def load(properties: dict) -> KalpaTTSConfig:
    config = KalpaTTSConfig(**properties)
    config.update_params()
    return config


def test_package_import_registers_the_addon():
    from ten_packages.extension import kalpa_tts_python

    # Importing the package is what runs the registration decorator.
    assert kalpa_tts_python.addon.KalpaTTSExtensionAddon.__name__ == (
        "KalpaTTSExtensionAddon"
    )


def test_defaults_need_only_a_key():
    config = load({"params": {"api_key": KEY}})
    config.check()
    assert config.model == DEFAULT_MODEL
    assert config.voice_id == DEFAULT_VOICE_ID
    assert config.sample_rate == DEFAULT_SAMPLE_RATE
    assert (
        config.websocket_url
        == f"wss://api.kalpalabs.ai/v1/tts/{DEFAULT_VOICE_ID}/stream"
    )
    assert config.session_message() == {
        "type": "initializeConnection",
        "api_key": KEY,
        "model": DEFAULT_MODEL,
        "output_format": "pcm_s16le",
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "generation_config": {},
    }


def test_missing_key_or_voice_is_rejected():
    with pytest.raises(ValueError, match="api_key"):
        load({"params": {"api_key": ""}}).check()
    with pytest.raises(ValueError, match="voice_id"):
        load({"params": {"api_key": KEY}, "voice_id": ""}).check()


def test_empty_voice_and_model_in_params_fall_back_to_the_defaults():
    config = load({"params": {"api_key": KEY, "voice_id": "", "model": None}})
    config.check()
    assert config.voice_id == DEFAULT_VOICE_ID
    assert config.model == DEFAULT_MODEL
    assert config.params == {}


def test_every_sample_rate_kalpa_offers_is_accepted():
    for rate in SAMPLE_RATES:
        config = load({"params": {"api_key": KEY, "sample_rate": rate}})
        config.check()
        assert config.session_message()["sample_rate"] == rate


@pytest.mark.parametrize(
    "value", [True, "fast", 24000.5, 0, -24000, 12345, None, [24000]]
)
def test_any_other_sample_rate_is_rejected(value):
    with pytest.raises(ValueError, match="sample_rate"):
        load({"params": {"api_key": KEY, "sample_rate": value}}).check()
    with pytest.raises(ValueError, match="sample_rate"):
        load({"sample_rate": value, "params": {"api_key": KEY}}).check()


@pytest.mark.parametrize(
    "value", ["80", 80, [50, "soon"], [True], {"first": 50}]
)
def test_a_malformed_chunk_schedule_is_rejected(value):
    with pytest.raises(ValueError, match="chunk_length_schedule"):
        load({"params": {"api_key": KEY, "chunk_length_schedule": value}})


@pytest.mark.parametrize("value", [None, []])
def test_an_empty_chunk_schedule_leaves_the_choice_to_kalpa(value):
    config = load({"params": {"api_key": KEY, "chunk_length_schedule": value}})
    assert config.session_message()["generation_config"] == {}
    assert "params" not in config.session_message()


def test_session_settings_leave_params_and_the_rest_is_passed_through():
    config = load(
        {
            "params": {
                "api_key": KEY,
                "voice_id": "voice-1",
                "model": "kalpa-tts-2",
                "sample_rate": "16000",
                "chunk_length_schedule": [50, 120],
                "temperature": 0.7,
                "audio_quality": "medium",
                # The extension sets these itself.
                "output_format": "mp3",
                "type": "sendText",
            }
        }
    )
    assert config.params == {"temperature": 0.7, "audio_quality": "medium"}
    message = config.session_message()
    assert message["model"] == "kalpa-tts-2"
    assert message["sample_rate"] == 16000
    assert message["output_format"] == "pcm_s16le"
    assert message["generation_config"] == {"chunk_length_schedule": [50, 120]}
    assert message["params"] == {"temperature": 0.7, "audio_quality": "medium"}
    assert config.websocket_url.endswith("/v1/tts/voice-1/stream")


def test_endpoint_precedence():
    base = {"api_key": KEY, "voice_id": "v1"}
    nested = load(
        {
            "url": "wss://top.example/tts",
            "params": {
                **base,
                "url": "wss://nested.example/tts/{voice_id}",
                "base_url": "wss://base.example",
            },
        }
    )
    assert nested.websocket_url == "wss://nested.example/tts/v1"

    # A blank `params.url` is no endpoint.
    top = load(
        {
            "url": "wss://top.example/tts",
            "params": {**base, "url": " ", "base_url": "wss://base.example"},
        }
    )
    assert top.websocket_url == "wss://top.example/tts"

    from_base = load({"params": {**base, "base_url": "wss://base.example/"}})
    assert from_base.websocket_url == "wss://base.example/v1/tts/v1/stream"
    # The endpoint is never sent to Kalpa as a generation parameter.
    assert "params" not in nested.session_message()
    assert "params" not in top.session_message()


def test_logged_config_hides_the_key():
    config = load(
        {
            "url": f"wss://example.test/tts?api_key={KEY}",
            "params": {"api_key": KEY, "token": KEY},
        }
    )
    logged = config.to_str(sensitive_handling=True)
    assert KEY not in logged
    assert KEY in config.to_str(sensitive_handling=False)


def test_the_endpoint_is_logged_and_reported_without_credentials():
    config = load(
        {
            "url": f"wss://user:{KEY}@example.test:8443/tts/{{voice_id}}"
            f"?token={KEY}#{KEY}",
            "params": {"api_key": KEY},
        }
    )
    assert config.safe_url == f"wss://example.test:8443/tts/{DEFAULT_VOICE_ID}"
    assert KEY not in config.to_str(sensitive_handling=True)
    # The connection itself still uses the address as it was given.
    assert KEY in config.websocket_url

    local = load({"params": {"api_key": KEY, "base_url": "ws://[::1]:9000/"}})
    assert local.safe_url == f"ws://[::1]:9000/v1/tts/{DEFAULT_VOICE_ID}/stream"


def test_long_text_is_cut_at_a_space():
    assert _split("short", 10) == ["short"]
    assert _split("aaaa bbbb cccc", 10) == ["aaaa bbbb ", "cccc"]
    assert _split("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]


def test_error_classification():
    def error(error_type, **extra):
        return {"type": "error", "error": {"type": error_type}, **extra}

    for permanent in (
        "authentication_error",
        "account_suspended",
        "allowance_exhausted",
        "not_found",
    ):
        assert _error_from_message(error(permanent)).permanent
    # Bad session settings are refused on every retry; a rejected message
    # on an open session is not.
    assert _error_from_message(error("invalid_request"), opening=True).permanent
    assert not _error_from_message(error("invalid_request")).permanent
    failed = _error_from_message(error("inference_error", response_id="r1"))
    assert not failed.permanent
    assert failed.response_id == "r1"


@pytest.mark.parametrize(
    "code,error_type",
    [
        (4401, "authentication_error"),
        (4402, "allowance_exhausted"),
        (4403, "account_suspended"),
        (4404, "not_found"),
    ],
)
def test_close_codes_that_no_reconnect_can_fix(code, error_type):
    error = _error_from_close(ConnectionClosedError(Close(code, ""), None))
    assert error.permanent
    assert (error.error_type, error.code) == (error_type, code)


def test_other_close_codes_are_worth_a_reconnect():
    for code in (1011, 4429):
        error = _error_from_close(
            ConnectionClosedError(Close(code, "try again"), None)
        )
        assert not error.permanent
        assert (error.error_type, error.code) == ("connection_closed", code)
        assert error.message == "try again"
    # Session settings Kalpa refuses are refused again. The same code on
    # an open session is one rejected message.
    refused = ConnectionClosedError(Close(4400, "invalid_request"), None)
    assert _error_from_close(refused, opening=True).permanent
    assert not _error_from_close(refused).permanent
