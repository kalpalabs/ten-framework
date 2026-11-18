#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import threading

import pytest

from harness import (
    END_REASON_ERROR,
    END_REASON_INTERRUPTED,
    END_REASON_REQUEST_END,
    run,
)
from mock_server import AUDIO, CHUNKS


def test_flush_interrupts_and_the_next_request_gets_a_new_session(
    server, config
):
    # Kalpa keeps speaking the first request until the flush arrives.
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(
            ten_env, "r1", "This request is interrupted.", end=True
        )
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    ends = rec.of("tts_audio_end")
    assert [(e["request_id"], e["reason"]) for e in ends] == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    names = rec.names()
    assert names.index("tts_audio_end") < names.index("tts_flush_end")
    assert rec.of("tts_flush_end")[0]["flush_id"] == "flush"
    # Only the first chunk of r1 was heard, then all of r2.
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO
    # The interrupted session was dropped; r2 ran on a new one.
    assert len(server.sessions) == 2
    assert server.sessions[1]["utterances"] == ["This one completes."]
    assert not rec.errors()


def test_flush_before_any_audio_ends_the_request_without_an_audio_start(
    server, config
):
    server.first_audio_delay_s = 5.0

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never heard.", end=True)
        await asyncio.sleep(0.2)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        # Nothing of the interrupted request arrives afterwards.
        await asyncio.sleep(0.3)

    rec = run(config, script)

    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_INTERRUPTED
    assert end["request_event_interval_ms"] == 0
    assert end["request_total_audio_duration_ms"] == 0
    assert "tts_audio_start" not in rec.names()
    assert "pcm_frame" not in rec.names()
    assert rec.names().index("tts_audio_end") < rec.names().index(
        "tts_flush_end"
    )


def test_flush_drops_text_that_was_not_spoken(server, config):
    async def script(rec, ten_env):
        # No `text_input_end`: Kalpa is still waiting for the rest.
        await rec.send_text(ten_env, "r1", "Half a sentence")
        await asyncio.sleep(0.2)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "A new request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # The unfinished text never became speech, on either session.
    assert server.utterances == ["A new request."]
    assert bytes(rec.audio) == AUDIO


def test_connection_status_is_reported(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        await rec.send_text(ten_env, "r2", "Interrupted.", turn_id=2)
        await asyncio.sleep(0.1)
        await rec.send_flush(ten_env)
        await rec.wait_for(
            lambda rec: [
                s["current"] for s in rec.of("connection_status_changed")
            ][-1:]
            == ["connected"]
            and "tts_flush_end" in rec.names()
        )

    rec = run(config, script)

    statuses = rec.of("connection_status_changed")
    # A last `disconnected` may follow when the test stops the extension.
    assert [s["current"] for s in statuses][:5] == [
        "connecting",
        "connected",
        "disconnected",
        "connecting",
        "connected",
    ]
    assert all(s["module"] == "tts" for s in statuses)
    assert all(s["vendor_info"]["vendor"] == "kalpa" for s in statuses)


def test_a_flush_while_connecting_ends_the_request_and_the_next_one_runs(
    server, config
):
    # The session does not open while r1 waits for it.
    server.hold_session = True

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never sent.", end=True)
        await asyncio.sleep(0.2)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    first = rec.of("tts_audio_end")[0]
    assert first["request_event_interval_ms"] == 0
    assert first["request_total_audio_duration_ms"] == 0
    # r1 was waiting for its session and never reached Kalpa.
    assert server.utterances == ["This one completes."]
    assert bytes(rec.audio) == AUDIO
    assert not rec.errors()


def test_a_flush_while_a_send_is_blocked_ends_the_request(
    server, config, monkeypatch, running
):
    # r1's text is stuck on its way out when the flush comes. The sender
    # holds the send lock at that moment.
    blocked = threading.Event()
    client = running.kalpa_tts.KalpaTTSClient
    send = client.send_text
    stuck = [True]

    async def blocked_send(self, text, flush=False):
        if stuck and text.strip():
            stuck.clear()
            blocked.set()
            await asyncio.sleep(3600)
        await send(self, text, flush)

    monkeypatch.setattr(client, "send_text", blocked_send)

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Stuck on its way out.", end=True)
        await rec.wait_for(lambda rec: blocked.is_set())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # The flush did not wait for the send, and r2 got the lock and a new
    # session. Nothing of r1 reached Kalpa or came back.
    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    first = rec.of("tts_audio_end")[0]
    assert first["request_event_interval_ms"] == 0
    assert first["request_total_audio_duration_ms"] == 0
    assert rec.index("tts_audio_end") < rec.index("tts_flush_end")
    assert server.utterances == ["This one completes."]
    assert [e["request_id"] for e in rec.of("tts_audio_start")] == ["r2"]
    assert bytes(rec.audio) == AUDIO
    assert len(server.sessions) == 2
    assert not rec.errors()


def test_late_events_of_a_replaced_connection_do_not_reach_the_next_request(
    server, config, monkeypatch, running
):
    # Every client the extension starts, with the loop it runs on.
    clients = []
    kalpa_tts = running.kalpa_tts
    start = kalpa_tts.KalpaTTSClient.start

    def recorded_start(self):
        clients.append((self, asyncio.get_running_loop()))
        start(self)

    monkeypatch.setattr(kalpa_tts.KalpaTTSClient, "start", recorded_start)
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Interrupted.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        server.holds = [1]
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: len(rec.of("pcm_frame")) == 2)
        # What r1's connection could still have delivered while r2 is
        # speaking: more audio, the end of its response, an error and the
        # loss of the connection.
        old, loop = clients[0]
        late = kalpa_tts.KalpaTTSError("inference_error", "Too late.")
        for name, args in (
            ("on_audio", (b"\x7f\x00" * 480, 0.0)),
            ("on_response_done", ("completed",)),
            ("on_response_error", (late,)),
            ("on_connection_failed", (late, False)),
        ):
            deliver = getattr(old, "_emit")(name, *args)
            asyncio.run_coroutine_threadsafe(deliver, loop).result(5)
        await asyncio.sleep(0.2)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # None of it touched r2: no extra audio, no early end, no error.
    assert len(clients) == 2
    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO
    assert not rec.errors()


@pytest.mark.parametrize("waiting_for", ["session", "send"])
def test_stopping_while_a_request_is_blocked_stops_the_extension(
    server, config, monkeypatch, running, waiting_for
):
    blocked = threading.Event()
    if waiting_for == "session":
        server.hold_session = True
    else:

        async def blocked_send(self, text, flush=False):
            blocked.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(
            running.kalpa_tts.KalpaTTSClient, "send_text", blocked_send
        )

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never spoken.", end=True)
        if waiting_for == "send":
            await rec.wait_for(lambda rec: blocked.is_set())
        else:
            await rec.wait_for(
                lambda rec: server.sessions and server.sessions[0]["init"]
            )
            await asyncio.sleep(0.2)

    # The extension is stopped here. `run` returns once it has stopped.
    rec = run(config, script, timeout_s=10.0)

    assert "pcm_frame" not in rec.names()
    assert server.utterances == []
    assert not rec.errors()


def test_a_second_flush_leaves_the_new_session_alone(server, config):
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Interrupted twice.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: len(rec.of("tts_flush_end")) == 2)
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    # One session was dropped for r1. The one that replaced it served r2.
    assert len(server.sessions) == 2
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO
    assert not rec.errors()


def test_a_flush_after_a_request_completed_emits_nothing_for_it(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Heard in full.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "So is this one.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert rec.ends() == [
        ("r1", END_REASON_REQUEST_END),
        ("r2", END_REASON_REQUEST_END),
    ]
    # Nothing was interrupted, so the session was kept.
    assert len(server.sessions) == 1
    assert bytes(rec.audio) == AUDIO * 2


def test_a_failure_to_report_the_disconnect_does_not_stop_the_extension(
    server, config, monkeypatch, running
):
    extension = running.extension.KalpaTTSExtension
    report = extension.on_disconnected
    failures = [RuntimeError("the report was not sent")]

    async def failing_report(self, *args, **kwargs):
        if failures:
            raise failures.pop()
        await report(self, *args, **kwargs)

    monkeypatch.setattr(extension, "on_disconnected", failing_report)
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Interrupted.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # r1 still ended, and r2 found a session to run on.
    assert not failures
    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert len(server.sessions) == 2
    assert not rec.errors()


@pytest.mark.parametrize("final", [False, True])
def test_text_that_arrives_after_its_flush_is_not_spoken(server, config, final):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "The start of a reply,")
        await asyncio.sleep(0.2)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        # A piece of r1 that was on its way when the flush came.
        await rec.send_text(
            ten_env, "r1", " and a late piece of it.", end=final
        )
        await rec.send_text(
            ten_env, "r2", "The next request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # The late piece neither reopened r1 nor held r2 up.
    assert server.utterances == ["The next request."]
    assert rec.ends() == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert [e["request_id"] for e in rec.of("tts_audio_start")] == ["r2"]
    assert bytes(rec.audio) == AUDIO
    assert not rec.errors()


def test_a_flush_while_a_request_is_ending_waits_for_its_last_events(
    server, config, monkeypatch, running
):
    # r1 fails after some audio, and its last events take a while.
    ending = threading.Event()
    extension = running.extension.KalpaTTSExtension
    report = extension.on_disconnected

    async def slow_report(self, *args, **kwargs):
        ending.set()
        await asyncio.sleep(0.3)
        await report(self, *args, **kwargs)

    monkeypatch.setattr(extension, "on_disconnected", slow_report)
    server.holds = [1]
    server.fail_after_chunks = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Fails part-way.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        server.release()
        await rec.wait_for(lambda rec: ending.is_set())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "This one completes.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # r1 ends once, for its own reason, before the flush is reported done.
    assert rec.ends() == [
        ("r1", END_REASON_ERROR),
        ("r2", END_REASON_REQUEST_END),
    ]
    names = rec.names()
    assert names.index("tts_audio_end") < names.index("tts_flush_end")
    assert [e["id"] for e in rec.errors()] == ["r1"]
    assert bytes(rec.audio) == CHUNKS[0][:-1] + AUDIO


def test_stopping_while_a_request_is_ending_lets_its_last_events_out(
    server, config, monkeypatch, running
):
    ending = threading.Event()
    # The tester is gone by the time the extension stops, so what the
    # extension does then is recorded here.
    order = []
    extension = running.extension.KalpaTTSExtension
    report = extension.on_disconnected
    send_end = extension.send_tts_audio_end

    async def recorded_report(self, *args, **kwargs):
        if kwargs.get("message") == "stopped":
            order.append("stopped")
        else:
            # r1's last events take a while.
            ending.set()
            await asyncio.sleep(0.3)
        await report(self, *args, **kwargs)

    async def recorded_end(self, **kwargs):
        order.append((kwargs["request_id"], kwargs["reason"].value))
        await send_end(self, **kwargs)

    monkeypatch.setattr(extension, "on_disconnected", recorded_report)
    monkeypatch.setattr(extension, "send_tts_audio_end", recorded_end)
    server.holds = [1]
    server.fail_after_chunks = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Fails part-way.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        server.release()
        await rec.wait_for(lambda rec: ending.is_set())

    # The extension is stopped here, while r1's last events are on their way.
    run(config, script)

    # r1 ended once, before the extension reported that it had stopped.
    assert order == [("r1", END_REASON_ERROR), "stopped"]
