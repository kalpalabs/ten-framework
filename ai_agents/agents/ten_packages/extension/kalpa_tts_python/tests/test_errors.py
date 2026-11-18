#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import threading

from harness import (
    END_REASON_ERROR,
    END_REASON_REQUEST_END,
    FATAL_ERROR,
    NON_FATAL_ERROR,
    run,
)
from mock_server import API_KEY, AUDIO, CHUNKS

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


# Long enough to start the utterance before its flush (segmented mode).
LONG_TEXT = "This sentence is long enough to start speaking before the flush."


def _error_codes(errors):
    return [(e["id"], e["code"], e["vendor_info"]["code"]) for e in errors]


def test_an_overloaded_refusal_before_the_flush_is_sent_again(server, config):
    server.overload_responses = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", LONG_TEXT)
        await rec.send_text(ten_env, "r1", " And a little more.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [("r1", NON_FATAL_ERROR, "overloaded")]
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert bytes(rec.audio) == AUDIO
    # Kalpa ended the first connection at the refusal. The request was sent
    # again in full on a new one, after waiting about a second.
    first, second = server.sessions
    assert first["utterances"] == []
    assert second["utterances"] == [LONG_TEXT + " And a little more."]
    assert second["opened_at"] - first["opened_at"] >= 1.0


def test_an_overloaded_refusal_after_the_flush_is_sent_again(server, config):
    server.overload_responses = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "A short reply.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [("r1", NON_FATAL_ERROR, "overloaded")]
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert bytes(rec.audio) == AUDIO
    # Kalpa kept the connection and cancelled the response; the request
    # went out again on a new connection.
    assert [s["utterances"] for s in server.sessions] == [
        ["A short reply."],
        ["A short reply."],
    ]
    assert (
        server.sessions[1]["opened_at"] - server.sessions[0]["opened_at"] >= 1.0
    )


def test_a_request_refused_on_every_attempt_ends_with_an_error(server, config):
    server.overload_responses = 3

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "A short reply.", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"), timeout_s=20.0)

    rec = run(config, script, timeout_s=25.0)

    # Sent three times in all, then failed; no audio started for it.
    assert (
        _error_codes(rec.errors())
        == [("r1", NON_FATAL_ERROR, "overloaded")] * 3
    )
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [("r2", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO
    # The waits double: about one second, then two, then four.
    opened = [s["opened_at"] for s in server.sessions]
    gaps = [later - earlier for earlier, later in zip(opened, opened[1:])]
    assert len(gaps) == 3
    assert gaps[0] >= 1.0 and gaps[1] >= 2.0 and gaps[2] >= 4.0
    assert server.sessions[-1]["utterances"] == ["The next request."]


def test_a_response_that_never_starts_fails_its_request(
    server, config, monkeypatch, running
):
    monkeypatch.setattr(running.extension, "RESPONSE_TIMEOUT_S", 0.3)
    # Kalpa takes the text and then sends nothing.
    server.holds = [0]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never answered.", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [
        ("r1", NON_FATAL_ERROR, "response_timeout")
    ]
    # No audio started for r1, so it has no audio end. r2 was not held up.
    assert rec.ends() == [("r2", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO
    # r2 ran on a session that never held r1.
    assert [s["utterances"] for s in server.sessions] == [
        ["Never answered."],
        ["The next request."],
    ]


def test_a_response_that_stalls_after_some_audio_fails_its_request(
    server, config, monkeypatch, running
):
    monkeypatch.setattr(running.extension, "RESPONSE_TIMEOUT_S", 1.0)
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Stops part-way.", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [
        ("r1", NON_FATAL_ERROR, "response_timeout")
    ]
    assert rec.ends() == [
        ("r1", END_REASON_ERROR),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert rec.of("tts_audio_end")[0]["request_total_audio_duration_ms"] == 20
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO


def test_the_deadline_runs_only_while_kalpa_is_silent_and_owes_audio(
    server, config, monkeypatch, running
):
    monkeypatch.setattr(running.extension, "RESPONSE_TIMEOUT_S", 1.0)
    # The response takes 1.5 seconds, longer than the deadline, but each
    # chunk comes well within it.
    server.chunks = CHUNKS * 2
    server.chunk_delay_s = 0.25

    async def script(rec, ten_env):
        # Until the text ends, Kalpa may be waiting for the rest of it.
        await rec.send_text(ten_env, "r1", "Half a sentence")
        await asyncio.sleep(1.3)
        await rec.send_text(ten_env, "r1", " and the rest.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert not rec.errors()
    assert rec.ends() == [("r1", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO * 2


def test_late_input_waits_until_the_failed_request_has_ended(
    server, config, monkeypatch, running
):
    # r1 fails while its text is still arriving, and its last events take
    # a while: the session is replaced among them.
    ending = threading.Event()
    extension = running.extension.KalpaTTSExtension
    report = extension.on_disconnected

    async def slow_report(self, *args, **kwargs):
        ending.set()
        await asyncio.sleep(0.3)
        await report(self, *args, **kwargs)

    monkeypatch.setattr(extension, "on_disconnected", slow_report)
    server.fail_responses = 1
    sentence = "This sentence is repeated to fill more than one utterance. "

    async def script(rec, ten_env):
        # 8,850 characters: the first utterance ends, and fails, before
        # the request does.
        await rec.send_text(ten_env, "r1", sentence * 150)
        await rec.wait_for(lambda rec: ending.is_set())
        await rec.send_text(ten_env, "r1", " Late text for it.", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # r2 found the new session in place; it did not fail as well.
    assert _error_codes(rec.errors()) == [
        ("r1", NON_FATAL_ERROR, "inference_error")
    ]
    assert rec.ends() == [
        ("r1", END_REASON_ERROR),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert server.sessions[-1]["utterances"] == ["The next request."]
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO


def test_connections_that_keep_failing_are_given_up_on_until_a_new_request(
    server, config, monkeypatch, running
):
    monkeypatch.setattr(running.kalpa_tts, "RECONNECT_BASE_DELAY_S", 0.01)
    # r1 is waiting for its session when the connections start to fail.
    server.hold_session = True
    server.fail_sessions = 5

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never gets through.", end=True)
        await asyncio.sleep(0.2)
        server.release()
        await rec.wait_for(lambda rec: len(rec.errors()) == 4)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # r1 is sent again twice and fails at the third failed connection.
    # After the fifth the client stops trying. Nothing has to change for a
    # later connection to work, so that report is not fatal either.
    assert _error_codes(rec.errors()) == [
        ("r1", NON_FATAL_ERROR, "connection_closed"),
        ("r1", NON_FATAL_ERROR, "connection_closed"),
        ("r1", NON_FATAL_ERROR, "connection_closed"),
        ("", NON_FATAL_ERROR, "connection_closed"),
    ]
    # r1 had no audio, so no audio end. A new request is a reason to
    # connect again, and it completes.
    assert rec.ends() == [("r2", END_REASON_REQUEST_END)]
    assert len(server.sessions) == 6
    assert server.utterances == ["The next request."]
    assert bytes(rec.audio) == AUDIO


def test_a_client_that_was_given_up_on_cannot_fail_the_next_request(
    server, config, monkeypatch, running
):
    monkeypatch.setattr(running.kalpa_tts, "RECONNECT_BASE_DELAY_S", 0.01)
    # The client's last report is slow, so r1 arrives after the client gave
    # up and before it has said so.
    giving_up = threading.Event()
    extension = running.extension.KalpaTTSExtension
    report = extension.on_disconnected

    async def slow_report(self, *args, **kwargs):
        if self.client is not None and self.client.failed is not None:
            giving_up.set()
            await asyncio.sleep(0.3)
        await report(self, *args, **kwargs)

    monkeypatch.setattr(extension, "on_disconnected", slow_report)
    server.fail_sessions = 5
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: giving_up.is_set())
        await rec.send_text(ten_env, "r1", "Runs on a new client.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        # The old client's report would have arrived by now.
        await asyncio.sleep(0.5)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert not rec.errors()
    assert rec.ends() == [("r1", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO
    assert len(server.sessions) == 6


def test_a_used_up_allowance_is_fatal_and_not_retried(server, config):
    server.refuse_sessions = ("allowance_exhausted", 4402)

    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: len(rec.errors()) >= 2)

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [
        ("", FATAL_ERROR, "allowance_exhausted"),
        ("r1", FATAL_ERROR, "allowance_exhausted"),
    ]
    assert len(server.sessions) == 1
    assert "pcm_frame" not in rec.names()


def test_an_unsupported_sample_rate_is_fatal_before_any_connection(server):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())

    rec = run(
        {
            "params": {
                "api_key": API_KEY,
                "base_url": server.url,
                "sample_rate": 12345,
            }
        },
        script,
    )

    (error,) = rec.errors()
    assert error["code"] == FATAL_ERROR
    assert "sample_rate" in error["message"]
    assert server.sessions == []


def test_an_endpoint_that_is_not_a_websocket_address_is_fatal_at_startup(
    server,
):
    async def script(rec, ten_env):
        await rec.wait_for(lambda rec: rec.errors())
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: len(rec.errors()) >= 2)

    rec = run(
        {
            "params": {
                "api_key": API_KEY,
                "base_url": server.url.replace("ws://", "http://"),
            }
        },
        script,
    )

    # Once at startup and once for the request; nothing is retried.
    assert [(e["id"], e["code"]) for e in rec.errors()] == [
        ("", FATAL_ERROR),
        ("r1", FATAL_ERROR),
    ]
    assert all("endpoint" in e["message"] for e in rec.errors())
    assert server.sessions == []


def test_frames_that_are_not_kalpa_messages_are_ignored(server, config):
    # Text that is not JSON, JSON that is not an object, a binary frame,
    # audio that is not base64 and a message type the client does not know.
    server.noise = True

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert not rec.errors()
    assert rec.ends() == [("r1", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO


def test_a_response_cancelled_without_an_error_fails_its_request(
    server, config
):
    server.cancel_responses = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Cancelled by Kalpa.", end=True)
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert _error_codes(rec.errors()) == [
        ("r1", NON_FATAL_ERROR, "response_cancelled")
    ]
    assert rec.ends() == [("r2", END_REASON_REQUEST_END)]
    assert bytes(rec.audio) == AUDIO
