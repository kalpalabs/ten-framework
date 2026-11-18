#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import time

import pytest

from harness import END_REASON_INTERRUPTED, END_REASON_REQUEST_END, run
from mock_server import AUDIO, AUDIO_MS, CHUNKS, SAMPLE_RATE

# The whole samples of the first chunk: its last byte is half a sample.
FIRST_CHUNK = CHUNKS[0][:-1]


@pytest.fixture(params=[False, True], ids=["no_dump", "dump"])
def dumping(request, config, tmp_path):
    """With and without a dump: closing its file is one more moment in
    which a flush can meet a request that is ending."""
    config["dump"] = request.param
    config["dump_path"] = str(tmp_path)
    return request.param


def test_flush_closes_the_response_and_the_next_request_completes(
    server, config, dumping, tmp_path
):
    # Kalpa keeps speaking the first request until its response is closed.
    server.hold_after_chunks = 1

    async def script(rec, ten_env):
        # The last text of the request: nothing more is coming for it.
        await rec.send_text(
            ten_env, "r1", "This request is interrupted.", end=True
        )
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.wait_for(lambda rec: server.requests[0]["closed_early"])
        # Nothing of the interrupted request arrives afterwards.
        await asyncio.sleep(0.2)
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
    # The audio that went out for r1, and no more.
    assert ends[0]["request_total_audio_duration_ms"] == (
        len(FIRST_CHUNK) * 1000 // (2 * SAMPLE_RATE)
    )
    names = rec.names()
    assert names.index("tts_audio_end") < names.index("tts_flush_end")
    assert rec.of("tts_flush_end")[0]["flush_id"] == "flush"
    second_start = [i for i, n in enumerate(names) if n == "tts_audio_start"][1]
    assert "pcm_frame" not in names[names.index("tts_flush_end") : second_start]
    assert bytes(rec.audio) == FIRST_CHUNK + AUDIO
    assert server.texts == [
        "This request is interrupted.",
        "This one completes.",
    ]
    assert not rec.errors()
    # The interrupted response's connection was closed, which is what
    # stops Kalpa, and the next request went out on a new one.
    interrupted, completed = server.requests
    assert interrupted["closed_early"]
    assert not completed["closed_early"]
    assert (interrupted["connection"], completed["connection"]) == (1, 2)
    if dumping:
        # The dump of the interrupted request holds what was heard of it.
        assert (tmp_path / "kalpa_dump_r1.pcm").read_bytes() == FIRST_CHUNK
        assert (tmp_path / "kalpa_dump_r2.pcm").read_bytes() == AUDIO


def test_flush_before_the_first_byte_closes_the_request_at_once(server, config):
    server.hold_before_response = True
    flushed_at = []

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Never heard.", end=True)
        await rec.wait_for(lambda rec: len(server.requests) == 1)
        flushed_at.append(time.monotonic())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.wait_for(lambda rec: server.requests[0]["closed_early"])
        server.hold_before_response = False
        await rec.send_text(ten_env, "r2", "Heard.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # Kalpa saw the connection close right after the flush, long before
    # any timeout: closing it is what stops the generation.
    assert server.requests[0]["closed_at"] - flushed_at[0] < 1.0
    # No audio started for r1, so there is no audio to end.
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r2"]
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [("r2", END_REASON_REQUEST_END)]
    assert rec.index("tts_flush_end") < rec.index("tts_audio_start")
    assert bytes(rec.audio) == AUDIO
    assert len(rec.ttfb()) == 1
    assert not rec.errors()
    # The next request needed a new connection.
    assert [r["connection"] for r in server.requests] == [1, 2]


def test_flush_drops_the_text_that_was_not_spoken_yet(server, config):
    server.hold_after_chunks = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "The first piece,")
        # These wait behind the first piece.
        await rec.send_text(ten_env, "r1", " the second piece,")
        await rec.send_text(ten_env, "r1", " and the last.", end=True)
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        server.hold_after_chunks = None
        await rec.send_text(
            ten_env, "r2", "A new request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == ["The first piece,", "A new request."]
    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_INTERRUPTED),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert bytes(rec.audio) == FIRST_CHUNK + AUDIO


def test_flush_between_two_pieces_ends_the_request(server, config):
    async def script(rec, ten_env):
        # No `text_input_end`: the request is waiting for more text.
        await rec.send_text(ten_env, "r1", "Half a reply,")
        await rec.wait_for(lambda rec: len(rec.audio) == len(AUDIO))
        await asyncio.sleep(0.1)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(
            ten_env, "r2", "A new request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    first, second = rec.of("tts_audio_end")
    assert (first["request_id"], first["reason"]) == (
        "r1",
        END_REASON_INTERRUPTED,
    )
    assert first["request_total_audio_duration_ms"] == AUDIO_MS
    assert (second["request_id"], second["reason"]) == (
        "r2",
        END_REASON_REQUEST_END,
    )
    assert bytes(rec.audio) == AUDIO + AUDIO


@pytest.mark.usefixtures("dumping")
def test_flush_that_meets_the_end_of_a_request_sends_one_audio_end(
    server, config
):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Just finishing.", end=True)
        # As soon as the end is seen, while the request is still closing.
        while not rec.ended("r1"):
            await asyncio.sleep(0)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await rec.send_text(ten_env, "r2", "Next.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert [
        (e["request_id"], e["reason"]) for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_REQUEST_END),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert bytes(rec.audio) == AUDIO + AUDIO
    assert not rec.errors()


def test_flush_while_waiting_to_send_again_sends_nothing_more(server, config):
    # Kalpa asks to wait a second before the text is sent again.
    server.refuse(503, "overloaded", "Busy.", retry_after="1")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Refused for now.", end=True)
        await rec.wait_for(lambda rec: len(server.requests) == 1)
        await asyncio.sleep(0.1)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        # Past the time the text would have been sent again.
        await asyncio.sleep(1.2)
        assert len(server.requests) == 1
        await rec.send_text(ten_env, "r2", "Heard.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == ["Refused for now.", "Heard."]
    assert [e["request_id"] for e in rec.of("tts_audio_end")] == ["r2"]
    assert bytes(rec.audio) == AUDIO
    assert not rec.errors()
