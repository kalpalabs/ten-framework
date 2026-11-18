#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import os

from ten_packages.extension.kalpa_tts_python.config import (
    DEFAULT_MODEL,
    DEFAULT_VOICE_ID,
)

from harness import END_REASON_REQUEST_END, run
from mock_server import AUDIO, AUDIO_MS, CHUNKS, SAMPLE_RATE


def test_one_request_is_one_audio_lifecycle(server, config):
    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello there.")
        await rec.send_text(ten_env, "r1", " How are you?")
        await rec.send_text(ten_env, "r1", "", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    # The text reaches Kalpa as it was written, in one utterance. The space
    # after its last sentence went out before the request was known to end.
    assert server.utterances == ["Hello there. How are you? "]
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


def test_a_finished_sentence_is_spoken_before_the_next_piece_arrives(
    server, config
):
    # A TEN agent cuts a reply after each punctuation mark, and the space
    # after the mark starts the next piece. Kalpa speaks a sentence once
    # whitespace follows it.
    first = "This first sentence is long enough for Kalpa to start speaking."

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", first)
        await rec.wait_for(
            lambda rec: "pcm_frame" in rec.names(), timeout_s=3.0
        )
        await rec.send_text(ten_env, "r1", " The second one comes later.")
        await rec.send_text(ten_env, "r1", "", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert server.utterances == [first + " The second one comes later. "]
    assert bytes(rec.audio) == AUDIO
    assert rec.names().count("tts_audio_start") == 1
    assert len(rec.ttfb()) == 1
    (end,) = rec.of("tts_audio_end")
    assert end["reason"] == END_REASON_REQUEST_END
    assert end["request_total_audio_duration_ms"] == AUDIO_MS
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


def test_interleaved_requests_keep_their_own_text_and_audio(server, config):
    async def script(rec, ten_env):
        # All four arrive before the first request has been spoken.
        await rec.send_text(ten_env, "r1", "First request,")
        await rec.send_text(ten_env, "r2", "Second request,", turn_id=2)
        await rec.send_text(ten_env, "r1", " first ending.", end=True)
        await rec.send_text(
            ten_env, "r2", " second ending.", end=True, turn_id=2
        )

    rec = run(config, script, done=lambda rec: rec.ended("r2"))

    assert server.utterances == [
        "First request, first ending.",
        "Second request, second ending.",
    ]
    assert [e["request_id"] for e in rec.of("tts_audio_start")] == ["r1", "r2"]
    assert rec.ends() == [
        ("r1", END_REASON_REQUEST_END),
        ("r2", END_REASON_REQUEST_END),
    ]
    assert bytes(rec.audio) == AUDIO * 2
    assert not rec.errors()


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


def test_ttfb_runs_from_the_first_text_sent_to_the_first_audio(
    server, config, clock
):
    # The session is slow to open. The text waits for it, and that wait is
    # not part of TTFB.
    server.hold_session = True
    server.holds = [0]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)
        await rec.wait_for(
            lambda rec: server.sessions and server.sessions[0]["init"]
        )
        clock.advance(500)
        server.release()
        await rec.wait_for(lambda rec: server.utterances == ["Hello world."])
        clock.advance(125)
        server.release()
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert rec.ttfb() == [125]


def test_ttfb_is_reported_when_audio_arrives_before_the_send_returns(
    server, config, monkeypatch, running
):
    # On a slow socket the send of a text can still be waiting when the
    # audio it asked for arrives.
    client = running.kalpa_tts.KalpaTTSClient
    send = client.send_text

    async def slow_send(self, text, flush=False):
        await send(self, text, flush)
        if text.strip():
            await asyncio.sleep(0.2)

    monkeypatch.setattr(client, "send_text", slow_send)
    first = "This first sentence is long enough for Kalpa to start speaking."

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", first)
        await rec.send_text(ten_env, "r1", " And the rest.", end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))

    rec = run(config, script)

    assert len(rec.ttfb()) == 1
    assert bytes(rec.audio) == AUDIO
    assert not rec.errors()


def test_usage_is_reported_once_for_each_request(server, config):
    texts = ["Sent twice.", "Fails before audio.", "Heard in full."]
    # The first connection drops at r1's text, so it is sent again.
    server.drop_on_text = 1

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", texts[0], end=True)
        await rec.wait_for(lambda rec: rec.ended("r1"))
        server.fail_responses = 1
        await rec.send_text(ten_env, "r2", texts[1], end=True, turn_id=2)
        await rec.wait_for(lambda rec: len(rec.usage()) == 2)
        await rec.send_text(ten_env, "r3", texts[2], end=True, turn_id=3)
        await rec.wait_for(lambda rec: rec.ended("r3"))

    rec = run(config, script)

    # Each request reports its own characters once: not twice for the
    # text sent again, and not with the next request for the one that
    # failed before it had any audio.
    assert rec.usage() == [
        (turn, len(text)) for turn, text in enumerate(texts, start=1)
    ]


def test_events_report_the_endpoint_without_credentials(server, config):
    secret = "sk-live-not-a-real-key-0123456789"
    config = {
        "params": {
            **config["params"],
            "url": f"{server.url}/v1/tts/{{voice_id}}/stream?token={secret}",
        }
    }

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    rec = run(config, script, done=lambda rec: rec.ended("r1"))

    assert secret not in repr(rec.events)
    (end,) = rec.of("tts_audio_end")
    assert end["metadata"]["vendor_metadata"] == {
        "url": f"{server.url}/v1/tts/{DEFAULT_VOICE_ID}/stream",
        "model": DEFAULT_MODEL,
        "voice_id": DEFAULT_VOICE_ID,
    }


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


def test_stopping_mid_request_keeps_the_audio_already_dumped(
    server, config, tmp_path
):
    config = {**config, "dump": True, "dump_path": str(tmp_path)}
    server.holds = [1]

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Stopped part-way.", end=True)
        await rec.wait_for(lambda rec: len(rec.audio) == len(CHUNKS[0]) - 1)

    # The extension is stopped here, with r1 still open.
    rec = run(config, script)

    assert "tts_audio_end" not in rec.names()
    assert (tmp_path / "kalpa_dump_r1.pcm").read_bytes() == CHUNKS[0][:-1]


def test_no_dump_files_unless_enabled(server, config, tmp_path):
    config = {**config, "dump_path": str(tmp_path)}

    async def script(rec, ten_env):
        await rec.send_text(ten_env, "r1", "Hello world.", end=True)

    run(config, script, done=lambda rec: rec.ended("r1"))
    assert os.listdir(tmp_path) == []
