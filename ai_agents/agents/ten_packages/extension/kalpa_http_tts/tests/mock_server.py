#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""A local Kalpa TTS HTTP server with scripted failures.

It answers `POST /v1/tts/{voice_id}/stream` the way Kalpa does: a chunked
`audio/pcm` body that starts with the first audio, or a status code with a
JSON error. The tests run the extension's real client against it, so
nothing in the extension is patched. It runs on its own thread and event
loop.
"""
import asyncio
import json
import threading
import time
from typing import Any

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

REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",
    403: "Forbidden",
    404: "Not Found",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


class MockKalpaServer:
    def __init__(self) -> None:
        # One entry per request, in order: its path, headers, JSON body,
        # the number of the connection it came on, when it arrived, and
        # whether the client closed the connection before the answer ended.
        self.requests: list[dict[str, Any]] = []
        self.connections = 0

        # Scripted behaviour, set by a test before it runs.
        self.first_audio_delay_s = 0.0
        # The chunks spoken for a text; any other text gets CHUNKS.
        self.audio: dict[str, list[bytes]] = {}
        # Keep a response open after this many chunks.
        self.hold_after_chunks: int | None = None
        # Do not answer at all: the client waits for the first byte.
        self.hold_before_response = False
        # The next requests are refused with these `(status, type, message,
        # headers)`, one each.
        self.refusals: list[tuple[int, str, str, dict[str, str]]] = []
        # The next N connections are dropped when their request arrives.
        self.drop_before_response = 0
        # The next N responses lose their connection after the first chunk.
        self.drop_mid_audio = 0
        self.content_type = "audio/pcm"

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def texts(self) -> list[str]:
        return [request["body"]["text"] for request in self.requests]

    def refuse(
        self,
        status: int,
        error_type: str,
        message: str,
        times: int = 1,
        **headers: str,
    ) -> None:
        self.refusals += [(status, error_type, message, headers)] * times

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
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self._port = server.sockets[0].getsockname()[1]
        self._ready.set()
        await self._stop.wait()
        server.close()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        connection = self.connections
        try:
            # Requests follow one another on a kept-alive connection.
            while True:
                request = await self._read_request(reader)
                if request is None:
                    return
                request["connection"] = connection
                self.requests.append(request)
                if not await self._answer(request, reader, writer):
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    @staticmethod
    async def _read_request(
        reader: asyncio.StreamReader,
    ) -> dict[str, Any] | None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return None
        request_line, *header_lines = head.decode("latin-1").split("\r\n")
        headers = {}
        for line in header_lines:
            name, _, value = line.partition(":")
            if name:
                headers[name.strip().lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", 0)))
        return {
            "path": request_line.split(" ")[1],
            "headers": headers,
            "body": json.loads(body or b"{}"),
            "at": time.monotonic(),
            "closed_early": False,
        }

    async def _answer(
        self,
        request: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bool:
        """Answer one request. False when the connection is finished."""
        if self.drop_before_response > 0:
            self.drop_before_response -= 1
            writer.transport.abort()
            return False
        if self.hold_before_response:
            await self._until_closed(request, reader)
            return False
        if request["headers"].get("authorization") != f"Bearer {API_KEY}":
            await self._refuse(
                writer, 401, "authentication_error", "Invalid API key.", {}
            )
            return True
        if self.refusals:
            await self._refuse(writer, *self.refusals.pop(0))
            return True

        if self.first_audio_delay_s:
            await asyncio.sleep(self.first_audio_delay_s)
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            + f"content-type: {self.content_type}\r\n".encode()
            + b"transfer-encoding: chunked\r\n\r\n"
        )
        chunks = self.audio.get(request["body"].get("text"), CHUNKS)
        for index, chunk in enumerate(chunks):
            writer.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            await writer.drain()
            if index == 0 and self.drop_mid_audio > 0:
                self.drop_mid_audio -= 1
                await asyncio.sleep(0.05)
                # No last chunk: the body ends before it is complete.
                writer.transport.abort()
                return False
            if self.hold_after_chunks == index + 1:
                await self._until_closed(request, reader)
                return False
            await asyncio.sleep(0.005)
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        return True

    @staticmethod
    async def _refuse(
        writer: asyncio.StreamWriter,
        status: int,
        error_type: str,
        message: str,
        headers: dict[str, str],
    ) -> None:
        body = json.dumps(
            {
                "error": {
                    "type": error_type,
                    "message": message,
                    "request_id": "req_mock",
                }
            }
        ).encode()
        head = [
            f"HTTP/1.1 {status} {REASONS.get(status, 'Error')}",
            "content-type: application/json",
            f"content-length: {len(body)}",
        ] + [
            f"{name.replace('_', '-')}: {value}"
            for name, value in headers.items()
        ]
        writer.write("\r\n".join(head).encode() + b"\r\n\r\n" + body)
        await writer.drain()

    @staticmethod
    async def _until_closed(
        request: dict[str, Any], reader: asyncio.StreamReader
    ) -> None:
        """Wait for the client to close the connection, and note when."""
        try:
            await reader.read()
        except ConnectionError:
            pass
        request["closed_early"] = True
        request["closed_at"] = time.monotonic()
