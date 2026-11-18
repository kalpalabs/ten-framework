#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""Known PCM on a clock the test controls, and what `tts_audio_end` says.

See `scripted.py`: the extension, its client and the base class run
unchanged; Kalpa and the clocks of the base class are scripted, so every
duration and interval here is exact.
"""
import asyncio

import httpx
from ten_ai_base.helper import PCMWriter

from harness import (
    END_REASON_ERROR,
    END_REASON_INTERRUPTED,
    END_REASON_REQUEST_END,
    NON_FATAL_ERROR,
)
from scripted import (
    HUNDRED_MS,
    Harness,
    accounting,
    flush,
    pcm,
    text_input,
    until,
)


def test_success_counts_emitted_audio_and_time_since_audio_start():
    first, second, third = (
        pcm(HUNDRED_MS, 1),
        # 50 ms and half a sample; the other half opens the next chunk.
        pcm(HUNDRED_MS // 2 + 1, 2),
        pcm(HUNDRED_MS // 2 - 1, 3),
    )

    async def scenario() -> Harness:
        async with Harness(
            (
                "audio",
                ("advance", 250),
                ("pcm", first),
                ("advance", 125),
                ("pcm", second),
                ("pcm", third),
                ("advance", 125),
            )
        ) as harness:
            await harness.send(text_input("Hello there.", True))
            await harness.finished()
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    names = env.names()
    assert names.index("tts_audio_start") < names.index("frame")
    assert max(i for i, n in enumerate(names) if n == "frame") < names.index(
        "tts_audio_end"
    )
    # Whole samples only, and all of them.
    assert [len(frame["pcm"]) for frame in env.of("frame")] == [
        HUNDRED_MS,
        HUNDRED_MS // 2,
        HUNDRED_MS // 2,
    ]
    assert env.pcm() == first + second + third
    # 16-bit mono at 24 kHz, and as many samples as the bytes hold.
    assert {
        (f["sample_rate"], f["bytes_per_sample"], f["channels"])
        for f in env.of("frame")
    } == {(24000, 2, 1)}
    assert [f["samples"] for f in env.of("frame")] == [2400, 1200, 1200]
    # From the request to the first audio.
    assert env.ttfb() == [250]
    (end,) = harness.ends()
    # 200 ms of audio went out. The interval runs from the audio start,
    # so the 250 ms before it are not part of it.
    assert accounting(end) == (END_REASON_REQUEST_END, 200, 250)
    # The request's usage: its characters and the audio received for it.
    usage = env.usage()[0]
    assert usage["output_characters"] == len("Hello there.")
    assert usage["recv_audio_duration"] == 200
    assert not env.of("error")
    assert harness.kalpa.closed == 1


def test_pieces_of_one_request_add_up_and_keep_the_first_audio_start():
    one, two, three = (
        pcm(HUNDRED_MS, 1),
        # The last sample of these two answers never completes.
        pcm(HUNDRED_MS + 1, 2),
        pcm(HUNDRED_MS // 2 + 1, 3),
    )

    async def scenario() -> Harness:
        async with Harness(
            ("audio", ("advance", 125), ("pcm", one)),
            ("audio", ("advance", 250), ("pcm", two), ("advance", 125)),
            ("audio", ("pcm", three)),
        ) as harness:
            await harness.send(text_input("One,", False))
            await harness.send(text_input(" two,", False))
            await harness.send(text_input(" three.", True))
            await harness.finished()
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    assert harness.kalpa.texts == ["One,", " two,", " three."]
    assert env.names().count("tts_audio_start") == 1
    # Measured once, for the first piece.
    assert env.ttfb() == [125]
    (end,) = harness.ends()
    # 100 + 100 + 50 ms: neither half sample was sent or counted, and
    # the first one did not shift the audio after it.
    assert env.pcm() == one + two[:-1] + three[:-1]
    assert accounting(end) == (END_REASON_REQUEST_END, 250, 375)
    usage = env.usage()[0]
    assert usage["output_characters"] == len("One, two, three.")
    assert usage["recv_audio_duration"] == 250


def test_an_error_after_audio_reports_the_audio_that_went_out(tmp_path):
    heard = pcm(HUNDRED_MS, 1)

    async def scenario() -> Harness:
        async with Harness(
            (
                "audio",
                ("advance", 125),
                ("pcm", heard),
                ("advance", 250),
                ("cut",),
            ),
            ("audio", ("pcm", pcm(HUNDRED_MS // 2, 2)), ("advance", 125)),
            dump=True,
            dump_path=str(tmp_path),
        ) as harness:
            await harness.send(text_input("Cut off part-way.", True))
            await harness.finished()
            await harness.send(text_input("Heard in full.", True, "r2"))
            await harness.finished("r2")
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    # Not sent again: part of it was heard.
    assert harness.kalpa.texts == ["Cut off part-way.", "Heard in full."]
    (error,) = env.of("error")
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == "connection_error: the connection was cut"
    first, second = harness.ends()
    assert accounting(first) == (END_REASON_ERROR, 100, 250)
    # The next request starts from nothing.
    assert accounting(second) == (END_REASON_REQUEST_END, 50, 125)
    assert env.ttfb() == [125, 0]
    assert [
        (u["output_characters"], u["recv_audio_duration"])
        for u in env.usage()[:2]
    ] == [(len("Cut off part-way."), 100), (len("Heard in full."), 50)]
    assert env.usage()[1]["total_recv_audio_duration"] == 150
    assert (tmp_path / "kalpa_dump_r1.pcm").read_bytes() == heard
    assert harness.kalpa.closed == 2


def test_a_flush_after_audio_reports_the_audio_that_went_out(tmp_path):
    first, second = pcm(HUNDRED_MS, 1), pcm(HUNDRED_MS // 2, 2)

    async def scenario() -> Harness:
        async with Harness(
            (
                "audio",
                ("pcm", first),
                ("advance", 125),
                ("pcm", second),
                ("hold",),
                # Never reached: the response is closed while it holds.
                ("pcm", pcm(HUNDRED_MS, 9)),
            ),
            ("audio", ("pcm", pcm(HUNDRED_MS, 3)), ("advance", 125)),
            dump=True,
            dump_path=str(tmp_path),
        ) as harness:
            # The last text of the request.
            await harness.send(text_input("Interrupted.", True))
            await harness.kalpa.holding.wait()
            await until(lambda: len(harness.ten_env.pcm()) == 7200)
            harness.clock.advance(250)
            await harness.send(flush())
            # The response was closed, not read to its end.
            await until(lambda: harness.kalpa.closed == 1)
            await harness.send(text_input("Next.", True, "r2"))
            await harness.finished("r2")
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    first_end, second_end = harness.ends()
    assert first_end["request_id"] == "r1"
    assert accounting(first_end) == (END_REASON_INTERRUPTED, 150, 375)
    assert accounting(second_end) == (END_REASON_REQUEST_END, 100, 125)
    names = env.names()
    assert names.index("tts_audio_end") < names.index("tts_flush_end")
    # Nothing of r1 after the flush; then r2, whole.
    second_start = [i for i, n in enumerate(names) if n == "tts_audio_start"][1]
    assert "frame" not in names[names.index("tts_flush_end") : second_start]
    assert env.pcm() == first + second + pcm(HUNDRED_MS, 3)
    assert (tmp_path / "kalpa_dump_r1.pcm").read_bytes() == first + second
    assert not env.of("error")


def test_a_flush_before_any_audio_ends_nothing():
    async def scenario() -> Harness:
        async with Harness(
            ("audio", ("advance", 250), ("hold",)),
            ("audio", ("pcm", pcm(HUNDRED_MS, 1))),
        ) as harness:
            await harness.send(text_input("Never heard.", True))
            await harness.kalpa.holding.wait()
            await harness.send(flush())
            await until(lambda: harness.kalpa.closed == 1)
            await harness.send(text_input("Heard.", True, "r2"))
            await harness.finished("r2")
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    # No audio started for r1, so no audio end either.
    assert [s["request_id"] for s in env.of("tts_audio_start")] == ["r2"]
    (end,) = harness.ends()
    assert end["request_id"] == "r2"
    assert accounting(end) == (END_REASON_REQUEST_END, 100, 0)
    assert env.names().index("tts_flush_end") < env.names().index(
        "tts_audio_start"
    )
    assert env.pcm() == pcm(HUNDRED_MS, 1)


def test_a_request_without_audio_reports_zero_for_both():
    async def scenario() -> Harness:
        async with Harness(
            # Kalpa takes a while to refuse.
            ("refuse", 250, 502, "inference_error"),
            ("audio", ("advance", 250)),
        ) as harness:
            await harness.send(text_input("Fails.", True))
            await harness.finished()
            await harness.send(text_input("Silence.", True, "r2"))
            await harness.finished("r2")
            await harness.send(text_input("...", True, "r3"))
            await harness.finished("r3")
        return harness

    harness = asyncio.run(scenario())
    env = harness.ten_env

    # Only the first two were worth a request.
    assert harness.kalpa.texts == ["Fails.", "Silence."]
    assert [accounting(end) for end in harness.ends()] == [
        # The 250 ms it waited are not an interval of audio.
        (END_REASON_ERROR, 0, 0),
        (END_REASON_REQUEST_END, 0, 0),
        (END_REASON_REQUEST_END, 0, 0),
    ]
    assert "tts_audio_start" not in env.names()
    assert "frame" not in env.names()
    assert not env.ttfb()
    (error,) = env.of("error")
    assert (error["code"], error["id"]) == (NON_FATAL_ERROR, "r1")


def test_an_answer_that_does_not_come_is_given_up_on():
    async def scenario() -> Harness:
        async with Harness(
            ("raise", httpx.ReadTimeout("no answer")),
            ("audio", ("pcm", pcm(HUNDRED_MS, 1))),
        ) as harness:
            await harness.send(text_input("No answer.", True))
            await harness.finished()
            await harness.send(text_input("Answered.", True, "r2"))
            await harness.finished("r2")
        return harness

    harness = asyncio.run(scenario())

    # Not sent again: a second wait would only add to the silence.
    assert harness.kalpa.texts == ["No answer.", "Answered."]
    (error,) = harness.ten_env.of("error")
    assert error["code"] == NON_FATAL_ERROR
    assert error["message"] == "timeout: Kalpa did not answer in time"
    assert [end["reason"] for end in harness.ends()] == [
        END_REASON_ERROR,
        END_REASON_REQUEST_END,
    ]


def test_a_flush_while_a_finished_request_closes_its_dump_ends_it_once(
    tmp_path,
):
    release = asyncio.Event()
    closing = asyncio.Event()
    real_flush = PCMWriter.flush

    async def held_flush(writer) -> None:
        # The request has sent its `tts_audio_end` and waits here.
        closing.set()
        await release.wait()
        await real_flush(writer)

    async def scenario() -> Harness:
        async with Harness(
            ("audio", ("pcm", pcm(HUNDRED_MS, 1)), ("advance", 125)),
            ("audio", ("pcm", pcm(HUNDRED_MS // 2, 2))),
            dump=True,
            dump_path=str(tmp_path),
        ) as harness:
            await harness.send(text_input("Just finishing.", True))
            await closing.wait()
            try:
                assert [end["reason"] for end in harness.ends()] == [
                    END_REASON_REQUEST_END
                ]
                # The flush finds the request still open and goes all the
                # way through while the dump is being closed.
                await asyncio.wait_for(harness.send(flush()), 2)
                assert harness.ten_env.names()[-1] == "tts_flush_end"
            finally:
                release.set()
            await until(harness.idle)
            await harness.send(text_input("Next.", True, "r2"))
            await harness.finished("r2")
        return harness

    PCMWriter.flush = held_flush
    try:
        harness = asyncio.run(scenario())
    finally:
        PCMWriter.flush = real_flush

    # One end for r1, the one it sent itself, and no interruption.
    assert [(e["request_id"], accounting(e)) for e in harness.ends()] == [
        ("r1", (END_REASON_REQUEST_END, 100, 125)),
        ("r2", (END_REASON_REQUEST_END, 50, 0)),
    ]
    assert (tmp_path / "kalpa_dump_r1.pcm").read_bytes() == pcm(HUNDRED_MS, 1)
    assert (tmp_path / "kalpa_dump_r2.pcm").read_bytes() == pcm(
        HUNDRED_MS // 2, 2
    )
    assert not harness.errors()
