#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import os

from harness import END_REASON_REQUEST_END, run
from mock_server import AUDIO, AUDIO_MS, SAMPLE_RATE


def test_one_request_is_one_audio_lifecycle(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.")
        await rec.send_text(ten_env, "r1", " How are you?")
        await rec.send_text(ten_env, "r1", "", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    # The text reaches Kalpa as it was written, in one utterance.
    assert server.utterances == ["Hello there. How are you?"]
    assert bytes(rec.audio) == AUDIO
    assert {f["sample_rate"] for f in rec.of("pcm_frame")} == {SAMPLE_RATE}

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
    assert not rec.errors()


def test_empty_request_ends_without_reaching_kalpa(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "")
        await rec.send_text(ten_env, "r1", "  \n\t", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    end = rec.of("tts_audio_end")[0]
    assert end["reason"] == END_REASON_REQUEST_END
    assert end["request_event_interval_ms"] == 0
    assert end["request_total_audio_duration_ms"] == 0
    assert "pcm_frame" not in rec.names()
    assert not rec.errors()
    assert server.utterances == []


def test_requests_run_one_after_another(server, config):
    async def script(rec, ten_env):
        # The second request arrives while the first is still speaking.
        await rec.send_text(ten_env, "r1", "First request.", end=True)
        await rec.send_text(
            ten_env, "r2", "Second request.", end=True, turn_id=2
        )

    rec = run(config, script, done=lambda rec: rec.ended("r2"))

    assert server.utterances == ["First request.", "Second request."]
    assert [e["request_id"] for e in rec.of("tts_audio_start")] == ["r1", "r2"]
    assert [e["request_id"] for e in rec.of("tts_audio_end")] == ["r1", "r2"]
    assert bytes(rec.audio) == AUDIO * 2
    # Each request has its own TTFB.
    assert len(rec.ttfb()) == 2
    # Both requests used the session opened at startup.
    assert len(server.sessions) == 1


def test_a_request_longer_than_one_utterance_continues_in_the_next(
    server, config
):
    sentence = "This sentence is repeated to fill more than one utterance. "
    text = sentence * 150  # 8,850 characters; an utterance holds 8,000.

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", text, end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert len(server.utterances) == 2
    assert all(len(u) <= 8000 for u in server.utterances)
    assert "".join(server.utterances) == text
    # Kalpa finishing the first utterance does not end the request.
    assert rec.names().count("tts_audio_start") == 1
    assert rec.names().count("tts_audio_end") == 1
    assert bytes(rec.audio) == AUDIO * 2
    assert rec.of("tts_audio_end")[0]["request_total_audio_duration_ms"] == (
        2 * AUDIO_MS
    )


def test_ttfb_runs_from_the_first_text_sent_to_the_first_audio(server, config):
    # A slow session must not count: the text is sent once it is ready.
    server.session_delay_s = 0.4
    server.first_audio_delay_s = 0.15

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    (ttfb,) = rec.ttfb()
    assert 150 <= ttfb < 350


def test_dump_writes_one_file_per_request(server, config, tmp_path):
    config = {**config, "dump": True, "dump_path": str(tmp_path)}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "First request.", end=True)
        await rec.send_text(ten_env, "r2", "", end=True, turn_id=2)
        await rec.send_text(
            ten_env, "r3", "Third request.", end=True, turn_id=3
        )

    run(config, script, done=lambda rec: rec.ended("r3"))

    # The empty request has no audio and so no file.
    assert sorted(os.listdir(tmp_path)) == [
        "kalpa_dump_r1.pcm",
        "kalpa_dump_r3.pcm",
    ]
    for name in os.listdir(tmp_path):
        assert (tmp_path / name).read_bytes() == AUDIO


def test_no_dump_files_unless_enabled(server, config, tmp_path):
    config = {**config, "dump_path": str(tmp_path)}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    run(config, script, done=lambda rec: rec.ended("r1"))
    assert os.listdir(tmp_path) == []
