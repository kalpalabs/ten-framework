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
import re
import threading
import time
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

REFUSALS = {
    "authentication_error": "Invalid API key.",
    "allowance_exhausted": "The allowance of this account is used up.",
}

# A sentence Kalpa takes as complete: its closing mark, then whitespace. A
# mark at the very end of the text is not one, since the text may go on.
SENTENCE_END = re.compile(r"[.!?…।]+[)\]\"'”’»]*\s")


class MockKalpaServer:
    def __init__(self) -> None:
        # One entry per connection: when it opened, its `initializeConnection`
        # message, the messages after it, and the text of each utterance it
        # flushed.
        self.sessions: list[dict[str, Any]] = []

        # Scripted behaviour, set by a test before it runs.
        self.session_delay_s = 0.0
        # The next session is not created until `release()`.
        self.hold_session = False
        # The next N connections are closed before their session is created.
        self.fail_sessions = 0
        # Every session is refused with this error type and close code.
        self.refuse_sessions: tuple[str, int] | None = None
        self.first_audio_delay_s = 0.0
        # Each response starts with frames that are not Kalpa messages.
        self.noise = False
        # The audio of every response, and the pause between its chunks.
        self.chunks = CHUNKS
        self.chunk_delay_s = 0.005
        # The next response waits, once it has sent each of these numbers of
        # chunks, until `release()` or until its connection closes.
        self.holds: list[int] = []
        # The next N utterances fail after their flush.
        self.fail_responses = 0
        # The next N responses end as cancelled, with no error before it.
        self.cancel_responses = 0
        # The next response fails after this many chunks.
        self.fail_after_chunks: int | None = None
        # The next N connections drop when their first text arrives.
        self.drop_on_text = 0
        # The next N responses lose their connection after the first chunk.
        self.drop_mid_audio = 0
        # The next N utterances are refused as `overloaded` when they start.
        self.overload_responses = 0
        # With `generation_config` (segmented mode), an utterance starts once
        # its text reaches this many characters and holds a complete
        # sentence, or at its flush. Its first audio follows at once.
        self.segment_threshold = 50

        self._ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._released: asyncio.Event | None = None
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

    def release(self) -> None:
        """Let a response that is waiting at one of its `holds` go on."""
        self._loop.call_soon_threadsafe(self._released.set)

    def _serve(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._released = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handle(self, ws: Any) -> None:
        session: dict[str, Any] = {
            "opened_at": time.monotonic(),
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
            hold, self.hold_session = self.hold_session, False
            if hold and not await self._wait_for_release(ws):
                return
            if self.fail_sessions > 0:
                self.fail_sessions -= 1
                await ws.close(code=1011, reason="internal error")
                return
            refusal = self.refuse_sessions
            if refusal is None and init.get("api_key") != API_KEY:
                refusal = ("authentication_error", 4401)
            if refusal is not None:
                error_type, code = refusal
                await self._send(
                    ws,
                    type="error",
                    fatal=True,
                    error={
                        "type": error_type,
                        "message": REFUSALS[error_type],
                    },
                )
                await ws.close(code=code, reason=error_type)
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

            segmented = init.get("generation_config") is not None
            buffer = ""
            # The utterance that started before its flush, if any.
            started: str | None = None
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
                if (
                    segmented
                    and started is None
                    and len(buffer.strip()) >= self.segment_threshold
                    and SENTENCE_END.search(buffer)
                ):
                    started = f"resp_{next(self._ids)}"
                    if self._refuse():
                        # Before the flush, Kalpa cannot tell this
                        # utterance's text from the next one's, so it ends
                        # the connection.
                        await self._overloaded(ws, started, fatal=True)
                        await ws.close(code=1011, reason="utterance failed")
                        return
                    await self._created(ws, started)
                    # The complete sentences are spoken before the flush.
                    if self.first_audio_delay_s:
                        await asyncio.sleep(self.first_audio_delay_s)
                    await self._audio(ws, started, 0)
                if message.get("flush"):
                    utterance, buffer = buffer, ""
                    session["utterances"].append(utterance)
                    response_id, started = started, None
                    spoken = 0 if response_id is None else 1
                    if response_id is None:
                        response_id = f"resp_{next(self._ids)}"
                        if self._refuse():
                            # After the flush the connection stays open.
                            await self._overloaded(ws, response_id, fatal=False)
                            await self._created(ws, response_id)
                            await self._done(
                                ws, response_id, utterance, "cancelled"
                            )
                            continue
                        await self._created(ws, response_id)
                    await self._respond(ws, utterance, response_id, spoken)
        except ConnectionClosed:
            pass

    def _refuse(self) -> bool:
        if self.overload_responses <= 0:
            return False
        self.overload_responses -= 1
        return True

    async def _overloaded(self, ws: Any, response_id: str, fatal: bool) -> None:
        await self._send(
            ws,
            type="error",
            fatal=fatal,
            request_id=response_id,
            response_id=response_id,
            error={
                "type": "overloaded",
                "message": "The model is temporarily overloaded. "
                "Retry shortly.",
            },
        )

    async def _created(self, ws: Any, response_id: str) -> None:
        await self._send(
            ws,
            type="responseCreated",
            response_id=response_id,
            sample_rate=SAMPLE_RATE,
            audio_quality="high",
        )

    async def _audio(self, ws: Any, response_id: str, index: int) -> None:
        await self._send(
            ws,
            type="responseAudio",
            response_id=response_id,
            pcm_b64=base64.b64encode(self.chunks[index]).decode(),
            frame=index,
        )

    async def _respond(
        self, ws: Any, utterance: str, response_id: str, spoken: int = 0
    ) -> None:
        """Send the audio after the first `spoken` chunks, then the end."""
        if self.fail_responses > 0:
            self.fail_responses -= 1
            await self._fail(ws, response_id, utterance)
            return
        if self.cancel_responses > 0:
            self.cancel_responses -= 1
            await self._done(ws, response_id, utterance, "cancelled")
            return
        if self.noise:
            await ws.send("not JSON")
            await ws.send("[1, 2, 3]")
            await ws.send(b"\x00\x01\x02")
            await self._send(
                ws,
                type="responseAudio",
                response_id=response_id,
                pcm_b64="not base64!",
            )
            await self._send(ws, type="somethingNew", response_id=response_id)
        holds, self.holds = self.holds, []
        if self.first_audio_delay_s and not spoken:
            await asyncio.sleep(self.first_audio_delay_s)
        for index in range(spoken, len(self.chunks) + 1):
            if index in holds and not await self._wait_for_release(ws):
                return
            if index == self.fail_after_chunks:
                self.fail_after_chunks = None
                await self._fail(ws, response_id, utterance)
                return
            if index == len(self.chunks):
                break
            await self._audio(ws, response_id, index)
            if index == 0 and self.drop_mid_audio > 0:
                self.drop_mid_audio -= 1
                await asyncio.sleep(0.05)
                ws.transport.abort()
                return
            await asyncio.sleep(self.chunk_delay_s)
        await self._done(ws, response_id, utterance, "completed")

    async def _wait_for_release(self, ws: Any) -> bool:
        """False when the connection closed before `release()`."""
        released = asyncio.ensure_future(self._released.wait())
        closed = asyncio.ensure_future(ws.wait_closed())
        await asyncio.wait(
            {released, closed}, return_when=asyncio.FIRST_COMPLETED
        )
        for waiter in (released, closed):
            waiter.cancel()
        if not self._released.is_set():
            return False
        self._released.clear()
        return True

    async def _fail(self, ws: Any, response_id: str, utterance: str) -> None:
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
