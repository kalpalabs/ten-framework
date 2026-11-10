#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio

from harness import END_REASON_INTERRUPTED, END_REASON_REQUEST_END, run
from mock_server import AUDIO, CHUNKS


def test_flush_interrupts_and_the_next_request_gets_a_new_session(
    server, config
):
    # Kalpa keeps speaking the first request until the flush arrives.
    server.hold_after_chunks = 1

    async def script(rec, ten_env):
        await rec.send_text(
            ten_env, "r1", "This request is interrupted.", end=True
        )
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        server.hold_after_chunks = None
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
