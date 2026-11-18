#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import os

from harness import END_REASON_REQUEST_END, run
from mock_server import API_KEY, AUDIO, AUDIO_MS, SAMPLE_RATE

VOICE = "18c26f49-9f9c-400c-bec9-7467f28e1db5"


def pcm(samples: int, seed: int) -> bytes:
    """Distinct audio per text, so mixed-up audio cannot compare equal."""
    return bytes((seed + index) % 251 for index in range(2 * samples))


def test_one_text_is_one_request_and_one_audio_lifecycle(server, config):
    config["params"]["temperature"] = 0.7
    # A setting of the WebSocket endpoint: accepted, and not sent.
    config["params"]["chunk_length_schedule"] = [50, 120]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    (request,) = server.requests
    assert request["path"] == f"/v1/tts/{VOICE}/stream"
    assert request["headers"]["authorization"] == f"Bearer {API_KEY}"
    # The key is in the header only; generation parameters ride `params`.
    assert request["body"] == {
        "text": "Hello there.",
        "model": "kalpa-tts-1",
        "output_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "params": {"temperature": 0.7},
    }

    # The two chunks that split a sample arrive as whole samples.
    assert bytes(rec.audio) == AUDIO
    frames = rec.of("pcm_frame")
    assert all(frame["bytes"] % 2 == 0 for frame in frames)
    # 16-bit mono at the configured rate, as the request asked of Kalpa.
    assert {
        (f["sample_rate"], f["bytes_per_sample"], f["channels"]) for f in frames
    } == {(SAMPLE_RATE, 2, 1)}

    names = rec.names()
    assert names.count("tts_audio_start") == 1
    assert names.count("tts_audio_end") == 1
    assert rec.index("tts_audio_start") < rec.index("pcm_frame")
    assert len(rec.ttfb()) == 1
    ttfb_at = next(
        i
        for i, (n, p) in enumerate(rec.events)
        if "ttfb" in p.get("metrics", {})
    )
    assert ttfb_at < rec.index("tts_audio_end")
    last_frame = max(i for i, n in enumerate(names) if n == "pcm_frame")
    assert last_frame < rec.index("tts_audio_end")

    end = rec.of("tts_audio_end")[0]
    assert end["request_id"] == "r1"
    assert end["reason"] == END_REASON_REQUEST_END
    assert end["request_total_audio_duration_ms"] == AUDIO_MS
    assert end["metadata"]["session_id"] == "session"
    assert end["metadata"]["turn_id"] == 1
    assert end["metadata"]["vendor_metadata"] == {
        "url": f"{server.url}/v1/tts/{VOICE}/stream",
        "model": "kalpa-tts-1",
        "voice_id": VOICE,
    }
    assert not rec.errors()


def test_each_piece_of_a_request_is_spoken_as_it_arrives(server, config):
    server.audio = {
        "Hello there,": [pcm(480, 1)],
        " how are you?": [pcm(240, 2), pcm(240, 3)],
        " Good.": [pcm(480, 4)],
    }

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there,")
        # The first piece is spoken before the next one exists.
        await rec.wait_for(lambda rec: "pcm_frame" in rec.names())
        await rec.send_text(ten_env, "r1", " how are you?")
        await rec.send_text(ten_env, "r1", " Good.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    # One request to Kalpa per piece, the text as it was written.
    assert server.texts == ["Hello there,", " how are you?", " Good."]
    assert bytes(rec.audio) == b"".join(
        chunk for chunks in server.audio.values() for chunk in chunks
    )
    # One audio lifecycle and one TTFB for the whole request.
    names = rec.names()
    assert names.count("tts_audio_start") == 1
    assert names.count("tts_audio_end") == 1
    assert len(rec.ttfb()) == 1
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert end["request_total_audio_duration_ms"] == (
        len(rec.audio) * 1000 // (2 * SAMPLE_RATE)
    )
    assert end["request_event_interval_ms"] > 0
    # The connection is kept between the requests.
    assert {request["connection"] for request in server.requests} == {1}
    assert not rec.errors()


def test_a_piece_with_nothing_to_say_is_not_sent(server, config):
    async def script(rec, ten_env):
        for text in ("", "  \n\t", "...", " - ", "\U0001f60a", "Hello."):
            await rec.send_text(ten_env, "r1", text)
        await rec.send_text(ten_env, "r1", "", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert server.texts == ["Hello."]
    assert bytes(rec.audio) == AUDIO
    assert rec.names().count("tts_audio_end") == 1
    assert not rec.errors()


def test_a_request_with_nothing_to_say_ends_without_audio(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "   ")
        await rec.send_text(ten_env, "r1", "", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        await rec.send_text(ten_env, "r2", "Hello.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    first, second = rec.of("tts_audio_end")
    assert first["request_id"] == "r1"
    assert first["reason"] == END_REASON_REQUEST_END
    assert first["request_event_interval_ms"] == 0
    assert first["request_total_audio_duration_ms"] == 0
    assert rec.index("tts_audio_end") < rec.index("tts_audio_start")
    assert second["request_id"] == "r2"
    assert server.texts == ["Hello."]
    assert bytes(rec.audio) == AUDIO
    assert not rec.errors()


def test_five_requests_in_a_row_keep_their_own_audio_and_usage(server, config):
    texts = ["One.", "Two two.", "Three three three.", "Four.", "Five five."]
    server.audio = {
        text: [pcm(240 * (index + 1), 40 + index)]
        for index, text in enumerate(texts)
    }

    async def script(rec, ten_env):
        for index, text in enumerate(texts):
            await rec.send_text(
                ten_env, f"r{index}", text, end=True, turn_id=index
            )
        await rec.wait_for(lambda rec: rec.ended("r4"))

    rec = run(config, script)

    assert server.texts == texts
    assert bytes(rec.audio) == b"".join(server.audio[t][0] for t in texts)
    ends = rec.of("tts_audio_end")
    assert [e["request_id"] for e in ends] == [f"r{i}" for i in range(5)]
    assert [e["metadata"]["turn_id"] for e in ends] == list(range(5))
    # 240 samples are 10 ms at 24 kHz.
    assert [e["request_total_audio_duration_ms"] for e in ends] == [
        10,
        20,
        30,
        40,
        50,
    ]
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == [
        f"r{i}" for i in range(5)
    ]
    assert len(rec.ttfb()) == 5
    # Each request reports its own characters and audio; the totals grow.
    usage = rec.usage()[:5]
    assert [u["output_characters"] for u in usage] == [len(t) for t in texts]
    assert [u["recv_audio_duration"] for u in usage] == [10, 20, 30, 40, 50]
    assert [u["total_output_characters"] for u in usage] == [
        sum(len(t) for t in texts[: i + 1]) for i in range(5)
    ]
    assert [u["total_recv_audio_duration"] for u in usage] == [
        10,
        30,
        60,
        100,
        150,
    ]
    # One connection served them all.
    assert {request["connection"] for request in server.requests} == {1}
    assert not rec.errors()


def test_a_request_that_arrives_inside_another_waits_for_its_end(
    server, config
):
    server.audio = {
        "First of one,": [pcm(480, 51)],
        " last of one.": [pcm(240, 52)],
        "All of two.": [pcm(960, 53)],
    }

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "First of one,")
        # r2 arrives whole before r1 has its last text.
        await rec.send_text(ten_env, "r2", "All of two.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: len(rec.audio) == 960)
        await rec.send_text(ten_env, "r1", " last of one.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    # All of r1, then r2.
    assert server.texts == ["First of one,", " last of one.", "All of two."]
    assert bytes(rec.audio) == pcm(480, 51) + pcm(240, 52) + pcm(960, 53)
    assert [
        (e["request_id"], e["reason"], e["request_total_audio_duration_ms"])
        for e in rec.of("tts_audio_end")
    ] == [
        ("r1", END_REASON_REQUEST_END, 30),
        ("r2", END_REASON_REQUEST_END, 40),
    ]
    names = rec.names()
    starts = [i for i, n in enumerate(names) if n == "tts_audio_start"]
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r1", "r2"]
    # r1 has ended before anything of r2 goes out.
    assert names.index("tts_audio_end") < starts[1]
    assert [e["metadata"]["turn_id"] for e in rec.of("tts_audio_end")] == [1, 2]
    assert [u["output_characters"] for u in rec.usage()[:2]] == [26, 11]
    assert not rec.errors()


def test_requests_follow_one_another_without_sharing_audio(server, config):
    server.audio = {"First.": [pcm(480, 11)], "Second.": [pcm(960, 12)]}

    async def script(rec, ten_env):
        # The second request arrives while the first is being spoken.
        await rec.send_text(ten_env, "r1", "First.", end=True)
        await rec.send_text(ten_env, "r2", "Second.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    rec = run(config, script)

    assert server.texts == ["First.", "Second."]
    assert bytes(rec.audio) == pcm(480, 11) + pcm(960, 12)
    ends = rec.of("tts_audio_end")
    assert [
        (e["request_id"], e["reason"], e["request_total_audio_duration_ms"])
        for e in ends
    ] == [
        ("r1", END_REASON_REQUEST_END, 20),
        ("r2", END_REASON_REQUEST_END, 40),
    ]
    assert [e["metadata"]["turn_id"] for e in ends] == [1, 2]
    assert [s["request_id"] for s in rec.of("tts_audio_start")] == ["r1", "r2"]
    assert len(rec.ttfb()) == 2
    # All of r1 is out before r2 starts.
    starts = [i for i, n in enumerate(rec.names()) if n == "tts_audio_start"]
    assert rec.index("tts_audio_end") < starts[1]


def test_a_sample_cut_off_at_the_end_is_neither_sent_nor_counted(
    server, config
):
    # 480 samples and half of another one.
    server.audio = {"Odd.": [pcm(240, 21) + b"\x07", b"\x08" + pcm(239, 22)]}
    server.audio["Odd."].append(b"\x09")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Odd.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    sent = b"".join(server.audio["Odd."])
    assert len(sent) == 961
    assert bytes(rec.audio) == sent[:960]
    (end,) = rec.of("tts_audio_end")
    assert end["request_total_audio_duration_ms"] == 20


def test_the_sample_rate_is_asked_of_kalpa_and_declared_on_the_frames(
    server, config
):
    config["params"]["sample_rate"] = 16000

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert server.requests[0]["body"]["sample_rate"] == 16000
    assert {f["sample_rate"] for f in rec.of("pcm_frame")} == {16000}
    (end,) = rec.of("tts_audio_end")
    assert end["request_total_audio_duration_ms"] == (
        len(AUDIO) * 1000 // (2 * 16000)
    )


def test_a_text_longer_than_one_request_is_spoken_in_two(server, config):
    words = "word " * 1700  # 8,500 characters
    # The cut leaves a third piece with nothing to say, which is not sent.
    tail = " " * 8000

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", words + tail, end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert [len(text) for text in server.texts] == [8000, 8000]
    assert "".join(server.texts) == words + tail[:7500]
    assert bytes(rec.audio) == AUDIO + AUDIO
    assert rec.names().count("tts_audio_end") == 1


def test_ttfb_runs_from_the_request_to_its_first_audio(server, config):
    server.first_audio_delay_s = 0.3

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    (ttfb,) = rec.ttfb()
    assert 300 <= ttfb < 600
    (end,) = rec.of("tts_audio_end")
    # The interval starts at the audio, not at the request.
    assert end["request_event_interval_ms"] < 250


def test_dump_holds_exactly_the_audio_of_each_request(server, config, tmp_path):
    server.audio = {"First.": [pcm(480, 31)], "Second.": [pcm(480, 32)]}
    config["dump"] = True
    config["dump_path"] = str(tmp_path / "dumps")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "First.", end=True)
        await rec.send_text(ten_env, "r2", "Second.", end=True, turn_id=2)
        await rec.wait_for(lambda rec: rec.ended("r2"))

    run(config, script)

    assert sorted(os.listdir(tmp_path / "dumps")) == [
        "kalpa_dump_r1.pcm",
        "kalpa_dump_r2.pcm",
    ]
    assert (tmp_path / "dumps" / "kalpa_dump_r1.pcm").read_bytes() == pcm(
        480, 31
    )
    assert (tmp_path / "dumps" / "kalpa_dump_r2.pcm").read_bytes() == pcm(
        480, 32
    )


def test_nothing_is_dumped_unless_asked(server, config, tmp_path):
    config["dump_path"] = str(tmp_path / "dumps")

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.", end=True)

    run(config, script, done=lambda rec: rec.ended("r1"))

    assert not (tmp_path / "dumps").exists()


def test_many_chunks_arrive_whole_and_in_order(server, config):
    # More chunks than the client reads ahead, each one sample and a half.
    chunks = [bytes([index % 251]) * 3 for index in range(100)]
    server.audio = {"Long.": chunks}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Long.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert bytes(rec.audio) == b"".join(chunks)
    assert all(frame["bytes"] % 2 == 0 for frame in rec.of("pcm_frame"))
    (end,) = rec.of("tts_audio_end")
    assert end["request_total_audio_duration_ms"] == 300 * 1000 // (
        2 * SAMPLE_RATE
    )
