#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""A local Kalpa TTS WebSocket server with scripted failures.

It speaks the protocol of `WSS /v1/tts/{voice_id}/stream`, so the tests run
the extension's real client against it and nothing in the extension is
patched. It runs on its own thread and event loop.
"""
import asyncio
import base64
import itertools
import json
import threading
from typing import Any

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

API_KEY = "test-key"
SAMPLE_RATE = 24000

# Three chunks. The first two split a 16-bit sample between them.
CHUNKS = [
    b"\x01\x00" * 480 + b"\x02",
    b"\x00" + b"\x03\x00" * 480,
    b"\x04\x00" * 480,
]
AUDIO = b"".join(CHUNKS)
AUDIO_MS = len(AUDIO) * 1000 // (2 * SAMPLE_RATE)


class MockKalpaServer:
    def __init__(self) -> None:
        # One entry per connection: its `initializeConnection` message, the
        # messages after it, and the text of each utterance it flushed.
        self.sessions: list[dict[str, Any]] = []

        # Scripted behaviour, set by a test before it runs.
        self.session_delay_s = 0.0
        self.first_audio_delay_s = 0.0
        # Keep a response open after this many chunks.
        self.hold_after_chunks: int | None = None
        # The next N utterances fail after their flush.
        self.fail_responses = 0
        # The next N connections drop when their first text arrives.
        self.drop_on_text = 0
        # The next N responses lose their connection after the first chunk.
        self.drop_mid_audio = 0

        self._ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}"

    @property
    def utterances(self) -> list[str]:
        return [text for s in self.sessions for text in s["utterances"]]

    def start(self) -> "MockKalpaServer":
        self._thread.start()
        self._ready.wait(5)
        return self

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)

    def _serve(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handle(self, ws: Any) -> None:
        session: dict[str, Any] = {
            "init": None,
            "messages": [],
            "utterances": [],
        }
        self.sessions.append(session)
        try:
            init = json.loads(await ws.recv())
            session["init"] = init
            if self.session_delay_s:
                await asyncio.sleep(self.session_delay_s)
            if init.get("api_key") != API_KEY:
                await self._send(
                    ws,
                    type="error",
                    fatal=True,
                    error={
                        "type": "authentication_error",
                        "message": "Invalid API key.",
                    },
                )
                await ws.close(code=4401, reason="authentication_error")
                return
            await self._send(
                ws,
                type="sessionCreated",
                session_id=f"sess_{len(self.sessions)}",
                model=init.get("model"),
                voice_id="voice",
                output_format="pcm_s16le",
                sample_rate=init.get("sample_rate", SAMPLE_RATE),
                channels=1,
                audio_quality="high",
            )

            buffer = ""
            async for raw in ws:
                message = json.loads(raw)
                session["messages"].append(message)
                if message.get("type") != "sendText":
                    continue
                text = message.get("text") or ""
                if text.strip() and self.drop_on_text > 0:
                    self.drop_on_text -= 1
                    ws.transport.abort()
                    return
                buffer += text
                if message.get("flush"):
                    utterance, buffer = buffer, ""
                    session["utterances"].append(utterance)
                    await self._respond(ws, utterance)
        except ConnectionClosed:
            pass

    async def _respond(self, ws: Any, utterance: str) -> None:
        response_id = f"resp_{next(self._ids)}"
        await self._send(
            ws,
            type="responseCreated",
            response_id=response_id,
            sample_rate=SAMPLE_RATE,
            audio_quality="high",
        )
        if self.fail_responses > 0:
            self.fail_responses -= 1
            await self._send(
                ws,
                type="error",
                fatal=False,
                response_id=response_id,
                error={
                    "type": "inference_error",
                    "message": "Generation failed.",
                },
            )
            await self._done(ws, response_id, utterance, "cancelled")
            return
        if self.first_audio_delay_s:
            await asyncio.sleep(self.first_audio_delay_s)
        for index, chunk in enumerate(CHUNKS):
            await self._send(
                ws,
                type="responseAudio",
                response_id=response_id,
                pcm_b64=base64.b64encode(chunk).decode(),
                frame=index,
            )
            if index == 0 and self.drop_mid_audio > 0:
                self.drop_mid_audio -= 1
                await asyncio.sleep(0.05)
                ws.transport.abort()
                return
            if self.hold_after_chunks == index + 1:
                await ws.wait_closed()
                return
            await asyncio.sleep(0.005)
        await self._done(ws, response_id, utterance, "completed")

    async def _done(
        self, ws: Any, response_id: str, utterance: str, status: str
    ) -> None:
        await self._send(
            ws,
            type="responseDone",
            response_id=response_id,
            status=status,
            text=utterance,
            usage={
                "input_chars": len(utterance),
                "input_audio_seconds": 0.0,
                "output_audio_seconds": AUDIO_MS / 1000,
            },
            meta={},
        )

    @staticmethod
    async def _send(ws: Any, **message: Any) -> None:
        await ws.send(json.dumps(message))
