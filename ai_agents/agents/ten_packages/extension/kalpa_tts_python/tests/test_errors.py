#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
from harness import (
    END_REASON_ERROR,
    END_REASON_REQUEST_END,
    FATAL_ERROR,
    NON_FATAL_ERROR,
    run,
)
from mock_server import AUDIO, CHUNKS

SECRET = "sk-live-not-a-real-key-0123456789"


def test_missing_key_is_fatal_at_startup(server):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())

    rec = run({"params": {"api_key": "", "base_url": server.url}}, script)

    (error,) = rec.errors()
    assert error["code"] == FATAL_ERROR
    assert "api_key" in error["message"]
    assert error["vendor_info"]["vendor"] == "kalpa"
    # Nothing was sent to Kalpa.
    assert server.sessions == []


def test_rejected_key_is_fatal_and_not_retried(server):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: len(rec.errors()) >= 2)

    rec = run({"params": {"api_key": SECRET, "base_url": server.url}}, script)

    # Once when the session is refused, once for the request it fails.
    assert [e["code"] for e in rec.errors()] == [FATAL_ERROR, FATAL_ERROR]
    for error in rec.errors():
        assert error["vendor_info"] == {
            "vendor": "kalpa",
            "code": "authentication_error",
            "message": "Invalid API key.",
        }
    assert rec.errors()[1]["id"] == "r1"
    assert len(server.sessions) == 1
    # The key is not echoed in anything the extension emits.
    assert SECRET not in repr(rec.events)
    assert "pcm_frame" not in rec.names()


def test_a_failed_response_ends_its_request_and_the_next_one_completes(
    server, config
):
    server.fail_responses = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "This one fails.", end=True)
        await rec.send_text(
            ten_env, "r2", "This one works.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert error["code"] == NON_FATAL_ERROR
    assert error["id"] == "r1"
    assert error["vendor_info"]["code"] == "inference_error"
    # No audio started for r1, so it has no audio end; r2 is whole.
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [("r2", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO
    # r2 ran on a session that never held r1.
    assert server.sessions[-1]["utterances"] == ["This one works."]


def test_a_request_is_sent_again_when_its_connection_drops_before_audio(
    server, config
):
    server.drop_on_text = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.")
        await rec.send_text(ten_env, "r1", " Still here.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    # The drop is reported, and the request still completes in full.
    assert [e["code"] for e in rec.errors()] == [NON_FATAL_ERROR]
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert bytes(rec.audio) == AUDIO
    assert len(rec.ttfb()) == 1
    assert server.sessions[-1]["utterances"] == ["Hello there. Still here."]
    statuses = [s["current"] for s in rec.of("connection_status_changed")]
    assert statuses[:5] == [
        "connecting",
        "connected",
        "disconnected",
        "connecting",
        "connected",
    ]


def test_a_drop_after_audio_started_fails_the_request(server, config):
    server.drop_mid_audio = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Cut off part-way.", end=True)
        await rec.send_text(
            ten_env, "r2", "Heard in full.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert [e["code"] for e in rec.errors()] == [NON_FATAL_ERROR]
    ends = rec.of("tts_audio_end")
    assert [(e["request_id"], e["reason"]) for e in ends] == [
        ("r1", END_REASON_ERROR),
        ("r2", END_REASON_REQUEST_END),
    ]
    # The error is reported before the audio end it explains.
    assert rec.index("error") < rec.index("tts_audio_end")
    # What was heard of r1 is not repeated.
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO
    assert server.sessions[-1]["utterances"] == ["Heard in full."]


def test_input_after_a_failure_does_not_block_the_next_request(server, config):
    # Every connection drops at the request's text, until it is given up on.
    server.drop_on_text = 3

    async def script(rec, ten_env):
        # r1 fails before its `text_input_end`; the rest of it still arrives.
        await rec.send_text(ten_env, "r1", "This request never gets through.")
        await rec.wait_for(lambda rec: len(rec.errors()) == 3)
        await rec.send_text(ten_env, "r1", " Late text for it.")
        await rec.send_text(ten_env, "r1", "", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # Two attempts to send it again, then the request fails for good.
    assert [e["code"] for e in rec.errors()] == [NON_FATAL_ERROR] * 3
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [("r2", END_REASON_REQUEST_END)]
    assert server.utterances == ["The next request."]
    assert bytes(rec.audio) == AUDIO
