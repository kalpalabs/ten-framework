#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""What `tts_audio_end` reports, checked against known audio and a clock
that the test moves.

The duration is that of the PCM sent downstream. The interval runs from
`tts_audio_start` to `tts_audio_end`. The server waits at its `holds`, so
the clock moves at known points of a request.
"""
import asyncio

import pytest

from harness import (
    END_REASON_ERROR,
    END_REASON_INTERRUPTED,
    END_REASON_REQUEST_END,
    NON_FATAL_ERROR,
    run,
)
from mock_server import AUDIO, AUDIO_MS, CHUNKS, SAMPLE_RATE

# The first chunk ends inside a sample; the byte after it never arrives when
# a request stops there.
FIRST = CHUNKS[0][:-1]


def duration_ms(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> int:
    """16-bit mono PCM."""
    return len(pcm) * 1000 // (sample_rate * 2)


def dumped(path, request_id: str) -> bytes:
    return (path / f"kalpa_dump_{request_id}.pcm").read_bytes()


def test_success_reports_the_emitted_audio_and_the_time_since_its_start(
    server, config, clock, tmp_path
):
    config = {**config, "dump": True, "dump_path": str(tmp_path)}
    # Wait before the first chunk and before the end of the response.
    server.holds = [0, len(CHUNKS)]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(lambda rec: server.utterances == ["Hello world."])
        clock.advance(125)
        server.release()
        await rec.wait_for(lambda rec: len(rec.audio) == len(AUDIO))
        clock.advance(250)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert bytes(rec.audio) == AUDIO
    for frame in rec.of("pcm_frame"):
        assert frame["sample_rate"] == SAMPLE_RATE
        assert frame["bytes_per_sample"] == 2
        assert frame["channels"] == 1
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    # The chunks split a sample between them; the bytes are summed first.
    assert end["request_total_audio_duration_ms"] == AUDIO_MS == 60
    # From the audio start to the audio end: not the 125 ms before it.
    assert end["request_event_interval_ms"] == 250
    assert rec.ttfb() == [125]
    assert dumped(tmp_path, "r1") == AUDIO
    assert not rec.errors()


def test_an_error_after_some_audio_reports_what_was_emitted(
    server, config, clock, tmp_path
):
    config = {**config, "dump": True, "dump_path": str(tmp_path)}
    server.holds = [1]
    server.fail_after_chunks = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Fails part-way.", end=True)
        await rec.wait_for(lambda rec: len(rec.audio) == len(FIRST))
        clock.advance(250)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r1"))
        await rec.send_text(
            ten_env, "r2", "Heard in full.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    (error,) = rec.errors()
    assert (error["id"], error["code"]) == ("r1", NON_FATAL_ERROR)
    assert error["vendor_info"]["code"] == "inference_error"
    first, second = rec.of("tts_audio_end")
    assert (first["request_id"], first["reason"]) == ("r1", END_REASON_ERROR)
    assert first["request_total_audio_duration_ms"] == duration_ms(FIRST) == 20
    assert first["request_event_interval_ms"] == 250
    assert rec.index("error") < rec.index("tts_audio_end")
    # The half sample r1 ended on is in neither the audio nor the dump, and
    # it does not shift r2.
    assert bytes(rec.audio) == FIRST + AUDIO
    assert dumped(tmp_path, "r1") == FIRST
    assert dumped(tmp_path, "r2") == AUDIO
    # r2 counts its own audio, from its own start.
    assert (second["request_id"], second["reason"]) == (
        "r2",
        END_REASON_REQUEST_END,
    )
    assert second["request_total_audio_duration_ms"] == AUDIO_MS
    assert second["request_event_interval_ms"] == 0


def test_a_flush_after_audio_started_reports_what_was_emitted(
    server, config, clock, tmp_path
):
    config = {**config, "dump": True, "dump_path": str(tmp_path)}
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Interrupted.", end=True)
        await rec.wait_for(lambda rec: len(rec.audio) == len(FIRST))
        clock.advance(375)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())

    rec = run(config, script)

    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_INTERRUPTED
    assert end["request_total_audio_duration_ms"] == duration_ms(FIRST) == 20
    assert end["request_event_interval_ms"] == 375
    assert bytes(rec.audio) == FIRST
    assert dumped(tmp_path, "r1") == FIRST
    assert not rec.errors()


def test_nothing_is_emitted_for_a_request_after_its_flush(server, config):
    # Kalpa keeps sending audio until the connection is gone.
    server.chunks = [b"\x05\x00" * 240] * 60
    server.chunk_delay_s = 0.01

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "A long reply.", end=True)
        await rec.wait_for(lambda rec: len(rec.of("pcm_frame")) >= 3)
        await rec.send_flush(ten_env)
        await rec.wait_for(lambda rec: "tts_flush_end" in rec.names())
        await asyncio.sleep(0.3)

    rec = run(config, script)

    names = rec.names()
    assert names.count("tts_audio_end") == 1
    last_frame = max(i for i, name in enumerate(names) if name == "pcm_frame")
    assert (
        last_frame < names.index("tts_audio_end") < names.index("tts_flush_end")
    )
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_INTERRUPTED
    assert 0 < len(rec.audio) < 60 * 480
    assert end["request_total_audio_duration_ms"] == duration_ms(rec.audio)


@pytest.mark.parametrize("sample_rate", [16000, 22050])
def test_only_whole_samples_are_sent_dumped_and_counted(
    server, config, tmp_path, sample_rate
):
    config = {
        "params": {**config["params"], "sample_rate": sample_rate},
        "dump": True,
        "dump_path": str(tmp_path),
    }
    # 45 bytes: the response ends inside a sample.
    server.chunks = [
        b"\x01\x00" * 10 + b"\x02",
        b"\x00" + b"\x03\x00" * 11 + b"\x04",
    ]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    emitted = b"".join(server.chunks)[:44]
    assert server.sessions[0]["init"]["sample_rate"] == sample_rate
    assert bytes(rec.audio) == emitted
    assert {f["sample_rate"] for f in rec.of("pcm_frame")} == {sample_rate}
    assert dumped(tmp_path, "r1") == emitted
    (end,) = rec.of("tts_audio_end")
    # 44 bytes are 1.375 ms at 16 kHz and 0.998 ms at 22.05 kHz.
    assert end["request_total_audio_duration_ms"] == duration_ms(
        emitted, sample_rate
    )
    assert end["request_total_audio_duration_ms"] == (
        1 if sample_rate == 16000 else 0
    )


def test_each_request_counts_its_own_audio(server, config, clock):
    server.holds = [len(CHUNKS)]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "First request.", end=True)
        await rec.wait_for(lambda rec: len(rec.audio) == len(AUDIO))
        clock.advance(500)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r1"))
        server.chunks = [b"\x06\x00" * 1200]
        await rec.send_text(
            ten_env, "r2", "Second request.", end=True, turn_id=2
        )
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    first, second = rec.of("tts_audio_end")
    assert first["request_total_audio_duration_ms"] == 60
    assert first["request_event_interval_ms"] == 500
    assert second["request_total_audio_duration_ms"] == 50
    assert second["request_event_interval_ms"] == 0
    assert bytes(rec.audio) == AUDIO + b"\x06\x00" * 1200
    # One TTFB each; the clock did not move while either request waited.
    assert rec.ttfb() == [0, 0]
