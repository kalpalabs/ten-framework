#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio

import pytest

from harness import (
    END_REASON_ERROR,
    END_REASON_REQUEST_END,
    FATAL_ERROR,
    NON_FATAL_ERROR,
    run,
)
from mock_server import AUDIO, CHUNKS, SAMPLE_RATE

from kalpa_http_tts import kalpa_tts

SECRET = "sk-live-not-a-real-key-0123456789"
FIRST_CHUNK = CHUNKS[0][:-1]


@pytest.fixture
def quick(monkeypatch):
    """Short waits before a text is sent again."""
    monkeypatch.setattr(kalpa_tts, "OVERLOADED_RETRY_DELAY_S", 0.05)
    monkeypatch.setattr(kalpa_tts, "RESEND_BASE_DELAY_S", 0.05)


def test_missing_key_is_fatal_at_startup_and_nothing_is_sent(server):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: len(rec.errors()) >= 2)

    rec = run({"params": {"api_key": "", "base_url": server.url}}, script)

    # Once at startup, once for the request that cannot be made.
    startup, request = rec.errors()
    for error in (startup, request):
        assert error["code"] == FATAL_ERROR
        assert error["message"] == (
            "Initialization failed: Required parameter 'api_key' is missing."
        )
        assert error["vendor_info"]["vendor"] == "kalpa"
    assert request["id"] == "r1"
    assert server.requests == []
    assert "pcm_frame" not in rec.names()


def test_a_key_with_a_line_break_is_fatal_at_startup_and_not_quoted(server):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    # As a key read from a file with its line end.
    rec = run(
        {"params": {"api_key": SECRET + "\n", "base_url": server.url}}, script
    )

    startup, request = rec.errors()
    for error in (startup, request):
        assert error["code"] == FATAL_ERROR
        assert error["message"] == (
            "Initialization failed: 'api_key' must not contain whitespace, "
            "control or non-ASCII characters."
        )
    assert request["id"] == "r1"
    assert server.requests == []
    assert SECRET not in repr(rec.events)


@pytest.mark.parametrize("rate", [0, 12345])
def test_a_rate_kalpa_does_not_offer_is_fatal_at_startup(server, config, rate):
    config["params"]["sample_rate"] = rate

    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    # The run ends: the extension stops without taking the runtime down.
    rec = run(config, script)

    startup, request = rec.errors()
    for error in (startup, request):
        assert error["code"] == FATAL_ERROR
        assert error["message"] == (
            "Initialization failed: 'sample_rate' must be one of "
            "8000, 16000, 22050, 24000, 32000, 44100, 48000."
        )
    assert server.requests == []
    assert "pcm_frame" not in rec.names()
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_ERROR
    assert end["request_total_audio_duration_ms"] == 0


def test_rejected_key_is_fatal_and_not_sent_again(server):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run({"params": {"api_key": SECRET, "base_url": server.url}}, script)

    (error,) = rec.errors()
    assert error["code"] == FATAL_ERROR
    assert error["id"] == "r1"
    assert error["message"] == "authentication_error: Invalid API key."
    assert error["vendor_info"]["vendor"] == "kalpa"
    assert len(server.requests) == 1
    # The key is not echoed in anything the extension emits.
    assert SECRET not in repr(rec.events)
    assert "pcm_frame" not in rec.names()


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (402, "allowance_exhausted"),
        (403, "account_suspended"),
        (404, "not_found"),
        (400, "invalid_request"),
        (422, "invalid_request"),
    ],
)
def test_a_refusal_that_needs_a_new_setting_is_fatal(
    server, config, status, error_type
):
    server.refuse(status, error_type, "Refused.")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert error["code"] == FATAL_ERROR
    assert error["message"] == f"{error_type}: Refused."
    assert len(server.requests) == 1
    assert "pcm_frame" not in rec.names()


def test_an_overloaded_refusal_is_sent_again_until_kalpa_takes_it(
    server, config
):
    server.refuse(503, "overloaded", "Busy.", times=2, retry_after="0.1")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    # The same text three times; the caller sees only the audio.
    assert server.texts == ["Hello there."] * 3
    assert not rec.errors()
    assert bytes(rec.audio) == AUDIO
    # The text is counted once, however often it was sent.
    assert rec.usage()[0]["output_characters"] == len("Hello there.")
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert len(rec.ttfb()) == 1
    # It waited for `Retry-After`, then twice as long.
    first, second, third = (request["at"] for request in server.requests)
    assert second - first >= 0.1
    assert third - second >= 0.2
    # The wait is part of the time to the first audio.
    assert rec.ttfb()[0] >= 300


def test_a_request_kalpa_keeps_refusing_as_overloaded_fails(
    server, config, quick
):
    server.refuse(503, "overloaded", "Busy.", times=3)

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never spoken.", end=True)
        await rec.send_text(ten_env, "r2", "Spoken.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == ["Never spoken."] * 3 + ["Spoken."]
    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["id"] == "r1"
    assert error["message"] == "overloaded: Busy."
    first, second = rec.of("tts_audio_end")
    assert (first["request_id"], first["reason"]) == ("r1", END_REASON_ERROR)
    assert first["request_event_interval_ms"] == 0
    assert first["request_total_audio_duration_ms"] == 0
    assert (second["request_id"], second["reason"]) == (
        "r2",
        END_REASON_REQUEST_END,
    )
    assert bytes(rec.audio) == AUDIO


def test_a_failed_generation_is_not_sent_again(server, config, quick):
    server.refuse(502, "inference_error", "The inference backend failed.")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "This one fails.", end=True)
        await rec.send_text(
            ten_env, "r2", "This one works.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == ["This one fails.", "This one works."]
    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["id"] == "r1"
    assert error["message"] == (
        "inference_error: The inference backend failed."
    )
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r2"]
    assert bytes(rec.audio) == AUDIO


def test_a_text_is_sent_again_when_its_connection_fails_before_audio(
    server, config, quick
):
    server.drop_before_response = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert server.texts == ["Hello there."] * 2
    assert not rec.errors()
    assert bytes(rec.audio) == AUDIO
    assert len(rec.ttfb()) == 1


def test_a_response_cut_off_after_audio_started_fails_the_request(
    server, config, quick
):
    server.drop_mid_audio = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Cut off part-way.", end=True)
        await rec.send_text(
            ten_env, "r2", "Heard in full.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # Not sent again: part of it was heard.
    assert server.texts == ["Cut off part-way.", "Heard in full."]
    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["id"] == "r1"
    assert error["message"].startswith("connection_error: ")
    first, second = rec.of("tts_audio_end")
    assert (first["request_id"], first["reason"]) == ("r1", END_REASON_ERROR)
    # The audio that went out before the cut, and only that.
    assert first["request_total_audio_duration_ms"] == (
        len(FIRST_CHUNK) * 1000 // (2 * SAMPLE_RATE)
    )
    assert (second["request_id"], second["reason"]) == (
        "r2",
        END_REASON_REQUEST_END,
    )
    assert rec.index("error") < rec.index("tts_audio_end")
    assert bytes(rec.audio) == FIRST_CHUNK + AUDIO


def test_a_request_goes_on_after_one_of_its_pieces_failed(server, config):
    server.refuse(502, "inference_error", "The inference backend failed.")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "This piece fails,")
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", " this one is spoken.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        # The request ended, so the next one is not held back.
        await rec.send_text(ten_env, "r2", "So is this.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["id"] == "r1"
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r1", "r2"]
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_REQUEST_END),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert bytes(rec.audio) == AUDIO + AUDIO


def test_a_request_waits_behind_one_kept_open_by_a_failed_piece(server, config):
    server.refuse(502, "inference_error", "The inference backend failed.")
    one, two = bytes([1, 0]) * 480, bytes([2, 0]) * 960
    server.audio = {" this one is spoken.": [one], "All of two.": [two]}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "This piece fails,")
        await rec.wait_for(lambda rec: rec.errors())
        # r1 is still open, so r2 waits for the rest of it.
        await rec.send_text(ten_env, "r2", "All of two.", end=True, turn_id=2)
        await asyncio.sleep(0.2)
        assert server.texts == ["This piece fails,"]
        await rec.send_text(ten_env, "r1", " this one is spoken.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == [
        "This piece fails,",
        " this one is spoken.",
        "All of two.",
    ]
    (error,) = rec.errors()
    assert (error["code"], error["id"]) == (NON_FATAL_ERROR, "r1")
    # r1's audio and end first, then r2 with its own audio and duration.
    assert bytes(rec.audio) == one + two
    assert [
        (e["request_id"], e["reason"], e["request_total_audio_duration_ms"])
        for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_REQUEST_END, 20),
        ("r2", END_REASON_REQUEST_END, 40),
    ]
    names = rec.names()
    starts = [i for i, n in enumerate(names) if n == "tts_audio_start"]
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r1", "r2"]
    assert names.index("tts_audio_end") < starts[1]


def test_a_request_id_used_again_is_a_new_request(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "First use.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        # A request in between that ends without a `tts_audio_end`.
        server.refuse(404, "not_found", "Unknown voice 'v'.")
        await rec.send_text(ten_env, "r2", "Never spoken,", turn_id=2)
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Second use.", end=True, turn_id=3)
        await rec.wait_for(lambda rec: len(rec.of("tts_audio_end")) == 2)
        await rec.send_text(ten_env, "r3", "After it.", end=True, turn_id=4)
        await rec.wait_for(lambda rec: rec.ended("r3"))

    rec = run(config, script)

    # The second r1 has its own start and end, and r3 is not held back.
    assert [
        (e["request_id"], e["reason"], e["metadata"]["turn_id"])
        for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_REQUEST_END, 1),
        ("r1", END_REASON_REQUEST_END, 3),
        ("r3", END_REASON_REQUEST_END, 4),
    ]
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == [
        "r1",
        "r1",
        "r3",
    ]
    assert bytes(rec.audio) == AUDIO * 3


def test_a_fatal_failure_before_the_last_text_releases_the_next_request(
    server, config
):
    server.refuse(404, "not_found", "Unknown voice 'v'.")

    async def script(rec, ten_env):
        # r2 arrives while r1, which has no `text_input_end` yet, is active.
        await rec.send_text(ten_env, "r1", "Never spoken,")
        await rec.send_text(ten_env, "r2", "Spoken.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert (error["code"], error["id"]) == (FATAL_ERROR, "r1")
    assert error["message"] == "not_found: Unknown voice 'v'."
    # r1 had no audio, so it has no audio end.
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [("r2", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO


def test_an_answer_that_is_not_audio_never_becomes_audio(server, config):
    server.content_type = "text/html"

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == (
        "unexpected_response: Kalpa answered with content type "
        "'text/html', not audio/pcm"
    )
    assert "pcm_frame" not in rec.names()
    assert "tts_audio_start" not in rec.names()


def test_an_empty_answer_ends_the_request_without_audio(server, config):
    server.audio = {"Silence.": []}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Silence.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert end["request_event_interval_ms"] == 0
    assert end["request_total_audio_duration_ms"] == 0
    assert "tts_audio_start" not in rec.names()
    assert not rec.ttfb()
    assert not rec.errors()


def test_an_answer_that_does_not_come_is_given_up_on(
    server, config, monkeypatch
):
    monkeypatch.setattr(kalpa_tts, "READ_TIMEOUT_S", 0.2)
    server.hold_before_response = True

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "No answer.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        await rec.wait_for(lambda rec: server.requests[0]["closed_early"])

    rec = run(config, script)

    # Not sent again: a second wait would only add to the silence.
    assert len(server.requests) == 1
    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == "timeout: Kalpa did not answer in time"
    assert "pcm_frame" not in rec.names()
