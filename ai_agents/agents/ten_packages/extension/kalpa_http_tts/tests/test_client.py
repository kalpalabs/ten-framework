#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""The client alone, read by a caller that falls behind or walks away."""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from ten_ai_base.struct import TTS2HttpResponseEventType as Event

from kalpa_http_tts import kalpa_tts
from kalpa_http_tts.config import KalpaHttpTTSConfig
from kalpa_http_tts.kalpa_tts import QUEUED_CHUNKS, KalpaHttpTTSClient

CHUNK = b"\x01\x00" * 10
CHUNKS = 200


class Body(httpx.AsyncByteStream):
    """Many small chunks, as fast as they are taken."""

    def __init__(self) -> None:
        self.taken = 0
        self.closed = False

    async def __aiter__(self):
        for _ in range(CHUNKS):
            self.taken += 1
            yield CHUNK
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        self.closed = True


def client_for(body: Body, options: dict | None = None) -> KalpaHttpTTSClient:
    """A client whose requests are answered with `body`. What it asks of
    the HTTP library is added to `options`."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "audio/pcm"}, stream=body
        )

    real_client = httpx.AsyncClient

    def scripted_client(**asked):
        if options is not None:
            options.update(asked)
        return real_client(
            transport=httpx.MockTransport(handle),
            headers=asked["headers"],
            timeout=asked["timeout"],
        )

    with patch.object(kalpa_tts.httpx, "AsyncClient", scripted_client):
        return KalpaHttpTTSClient(
            KalpaHttpTTSConfig(api_key="test-key"), MagicMock()
        )


def test_the_client_speaks_http_1_1():
    # Closing a response has to close its connection: that is what stops
    # Kalpa. Over HTTP/2 the HTTP library leaves the stream open.
    options: dict = {}

    async def scenario():
        await client_for(Body(), options).clean()

    asyncio.run(scenario())

    assert not options.get("http2")
    assert options.get("http1", True)
    assert options["headers"] == {"Authorization": "Bearer test-key"}
    assert options["timeout"].read == kalpa_tts.READ_TIMEOUT_S
    assert options["timeout"].connect == kalpa_tts.CONNECT_TIMEOUT_S
    # Nothing installs the HTTP/2 extra of the HTTP library either.
    package = Path(__file__).parents[1]
    for name in ("requirements.txt", "pyproject.toml"):
        assert "http2" not in (package / name).read_text()


def test_the_client_reads_only_a_little_ahead_of_its_caller():
    body = Body()

    async def scenario():
        client = client_for(body)
        assert client.get_extra_metadata() == {}
        events = client.get("Hello there.", "r1")
        first = await events.__anext__()
        # The caller is busy; Kalpa keeps sending.
        await asyncio.sleep(0.05)
        ahead = body.taken
        rest = [event async for event in events]
        await client.clean()
        return first, ahead, rest

    first, ahead, rest = asyncio.run(scenario())

    assert first == (CHUNK, Event.RESPONSE)
    # What the caller holds, a full queue, and the chunk waiting for room.
    assert ahead <= QUEUED_CHUNKS + 2
    # Nothing was lost by waiting.
    assert rest[-1] == (None, Event.END)
    assert rest[:-1] == [(CHUNK, Event.RESPONSE)] * (CHUNKS - 1)
    assert body.closed


def test_cancel_drops_the_audio_that_was_read_ahead():
    body = Body()

    async def scenario():
        client = client_for(body)
        events = client.get("Hello there.", "r1")
        first = await events.__anext__()
        await asyncio.sleep(0.05)
        await client.cancel()
        rest = [event async for event in events]
        # Cancelling again, or with nothing in flight, changes nothing.
        await client.cancel()
        await asyncio.sleep(0.05)
        await client.clean()
        return first, rest

    first, rest = asyncio.run(scenario())

    assert first == (CHUNK, Event.RESPONSE)
    assert rest == [(None, Event.FLUSH)]
    assert body.taken < CHUNKS
    assert body.closed


def test_a_caller_that_stops_reading_closes_the_response():
    body = Body()

    async def scenario():
        client = client_for(body)
        events = client.get("Hello there.", "r1")
        await events.__anext__()
        await events.aclose()
        await asyncio.sleep(0.05)
        closed, taken = body.closed, body.taken
        await client.clean()
        return closed, taken

    closed, taken = asyncio.run(scenario())

    assert closed
    assert taken < CHUNKS
