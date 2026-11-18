#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""Settings that fail at start, and secrets that must stay unsaid.

See `scripted.py`: the extension runs from `on_init` to `on_stop` on an
event loop of the test, so settings the test runtime would refuse to load
can be given to it, and every event and every log line can be searched.
"""
import asyncio

import httpx
import pytest

from kalpa_http_tts import kalpa_tts

from harness import END_REASON_ERROR, FATAL_ERROR, NON_FATAL_ERROR
from scripted import HUNDRED_MS, Harness, accounting, pcm, text_input

SECRET = "sk-live-not-a-real-key-0123456789"


def run(*answers, **properties) -> Harness:
    """Start the extension, send it one request, and stop it."""

    async def scenario() -> Harness:
        async with Harness(*answers, **properties) as harness:
            await harness.send(text_input("Hello world.", True))
            await harness.finished()
        return harness

    return asyncio.run(scenario())


def refused(harness: Harness, message: str) -> None:
    """The start failed with `message`, and so did the request: nothing
    was sent to Kalpa, and the extension stopped without an exception."""
    startup, request = harness.errors()
    assert (startup["code"], startup["id"]) == (FATAL_ERROR, "")
    assert (request["code"], request["id"]) == (FATAL_ERROR, "r1")
    assert startup["message"] == request["message"] == message
    assert harness.kalpa.texts == []
    assert "frame" not in harness.ten_env.names()
    assert [accounting(end) for end in harness.ends()] == [
        (END_REASON_ERROR, 0, 0)
    ]
    # The stop reported its usage, which divides by the sample rate.
    assert harness.ten_env.usage()[-1]["total_recv_audio_duration"] == 0


@pytest.mark.parametrize(
    "rate", [0, True, False, "fast", 12345, 24000.5, -24000, None]
)
def test_a_rate_kalpa_does_not_offer_fails_at_start_and_stops_cleanly(rate):
    harness = run(params={"api_key": "test-key", "sample_rate": rate})

    refused(
        harness,
        "Initialization failed: 'sample_rate' must be one of "
        "8000, 16000, 22050, 24000, 32000, 44100, 48000.",
    )
    # The rate the base class computes with is still a rate.
    assert harness.extension.synthesize_audio_sample_rate() == 24000


def test_a_missing_key_fails_at_start():
    harness = run(params={"api_key": ""})

    refused(
        harness,
        "Initialization failed: Required parameter 'api_key' is missing.",
    )


@pytest.mark.parametrize(
    "key",
    [
        SECRET + "\n",
        SECRET + "\r\n",
        SECRET + " ",
        "\t" + SECRET,
        SECRET[:10] + " " + SECRET[10:],
        SECRET + "\x7f",
        SECRET + "é",
    ],
)
def test_a_key_that_a_header_cannot_hold_fails_at_start_unquoted(key):
    harness = run(params={"api_key": key})

    refused(
        harness,
        "Initialization failed: 'api_key' must not contain whitespace, "
        "control or non-ASCII characters.",
    )
    # Neither the key nor the part of it before the bad character is in
    # any event or log line.
    assert SECRET not in harness.ten_env.said()
    assert SECRET[10:] not in harness.ten_env.said()
    # No header was built from it either.
    assert harness.http_options["headers"] == {}


def test_settings_that_cannot_be_read_fail_every_request_the_same_way():
    # The key has ended up where a table of settings belongs.
    harness = run(params=SECRET)

    message = harness.errors()[0]["message"]
    assert message.startswith("Initialization failed: params: Input should")
    # The request is refused for the same reason, not for a missing key.
    refused(harness, message)
    assert SECRET not in harness.ten_env.said()


def test_no_settings_at_all_fail_at_start():
    harness = Harness()
    harness.ten_env.get_property_to_json.return_value = ("{}", None)

    async def scenario() -> None:
        async with harness:
            await harness.send(text_input("Hello world.", True))
            await harness.finished()

    asyncio.run(scenario())

    startup, request = harness.errors()
    assert startup["code"] == request["code"] == FATAL_ERROR
    assert startup["message"].startswith(
        "Initialization failed: Configuration is empty."
    )
    assert request["message"] == (
        "Initialization failed: Configuration is empty."
    )
    assert harness.kalpa.texts == []


def test_what_the_http_library_says_about_a_request_is_not_repeated():
    # Its error for a header it will not send quotes the header.
    quoting = httpx.LocalProtocolError(
        f"Illegal header value b'Bearer {SECRET}\\n'"
    )
    harness = run(
        ("raise", quoting),
        ("audio", ("pcm", pcm(HUNDRED_MS, 1))),
        params={"api_key": SECRET},
    )

    # Not sent again: it would fail the same way.
    assert harness.kalpa.texts == ["Hello world."]
    (error,) = harness.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == (
        "request_error: The request could not be sent (LocalProtocolError)"
    )
    assert SECRET not in harness.ten_env.said()
    assert harness.http_options["headers"] == {
        "Authorization": f"Bearer {SECRET}"
    }


def test_an_unexpected_failure_is_named_by_its_kind_only():
    harness = run(
        ("raise", RuntimeError(f"cannot send {SECRET}")),
        params={"api_key": SECRET},
    )

    (error,) = harness.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == "internal_error: Unexpected RuntimeError"
    assert SECRET not in harness.ten_env.said()


def test_no_message_carries_the_key_whoever_worded_it(monkeypatch):
    monkeypatch.setattr(kalpa_tts, "RESEND_BASE_DELAY_S", 0.01)
    # A text of the network, which is reported, happens to hold the key.
    harness = run(
        *[("raise", httpx.ConnectError(f"refused {SECRET}"))] * 3,
        params={"api_key": SECRET},
    )

    assert harness.kalpa.texts == ["Hello world."] * 3
    (error,) = harness.errors()
    assert error["message"] == "connection_error: refused <api_key>"
    assert SECRET not in harness.ten_env.said()


def test_a_secret_in_the_endpoint_reaches_kalpa_and_nothing_else():
    token, password = "token-0123456789", "password-0123456789"
    harness = run(
        ("audio", ("pcm", pcm(HUNDRED_MS, 1))),
        params={
            "api_key": SECRET,
            "voice_id": "v1",
            "url": (
                f"https://user:{password}@kalpa.test:8443/speak/{{voice_id}}"
                f"?access={token}&api_key={token}#frag"
            ),
        },
    )

    # The request went to the endpoint as it was given.
    (url,) = harness.kalpa.urls
    assert password in url and token in url
    assert harness.ten_env.pcm() == pcm(HUNDRED_MS, 1)
    # Events carry it as scheme, host and path.
    metadata = [
        payload["metadata"]["vendor_metadata"]
        for _, payload in harness.ten_env.events
        if "vendor_metadata" in payload.get("metadata", {})
    ]
    assert metadata
    assert {m["url"] for m in metadata} == {"https://kalpa.test:8443/speak/v1"}
    said = harness.ten_env.said()
    assert token not in said and password not in said
    assert SECRET not in said
