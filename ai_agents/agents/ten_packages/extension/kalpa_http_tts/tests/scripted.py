#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""The extension on an event loop of the test, talking to a scripted Kalpa.

The extension, its client and the base class run unchanged, from `on_init`
to `on_stop`. Only Kalpa (an HTTP transport that answers from a script) and
the clocks of the base class are replaced, so every duration and interval
is exact, and everything the extension sends or logs can be read.
"""
import asyncio
from datetime import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from ten_runtime import Data

from kalpa_http_tts import kalpa_tts
from kalpa_http_tts.extension import KalpaHttpTTSExtension

# 24 kHz mono 16-bit PCM is 48 bytes per millisecond.
HUNDRED_MS = 4800


def pcm(size: int, seed: int) -> bytes:
    """Distinct bytes per chunk, so mixed-up audio cannot compare equal."""
    return bytes((seed + index) % 251 for index in range(size))


class FakeClock:
    """Moves only when a script says so, in steps (125 ms) binary floats
    hold exactly."""

    def __init__(self) -> None:
        self.at = 1000.0

    def monotonic(self) -> float:
        return self.at

    def time(self) -> float:
        return self.at

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.at)

    def advance(self, ms: int) -> None:
        self.at += ms / 1000


class RecordingTenEnv:
    """Hands the extension its properties and records what it sends and
    logs, in order."""

    def __init__(self, properties: dict) -> None:
        self.events: list[tuple[str, dict]] = []
        self.get_property_to_json = AsyncMock(
            return_value=(json.dumps(properties), None)
        )
        self.send_data = AsyncMock(side_effect=self._data)
        self.send_audio_frame = AsyncMock(side_effect=self._frame)
        self._logs = []
        for name in ("log_debug", "log_info", "log_warn", "log_error"):
            log = MagicMock()
            setattr(self, name, log)
            self._logs.append(log)

    async def _data(self, data) -> None:
        raw, _ = data.get_property_to_json("")
        self.events.append((data.get_name(), json.loads(raw)))

    async def _frame(self, frame) -> None:
        buffer = frame.lock_buf()
        try:
            audio = bytes(buffer)
        finally:
            frame.unlock_buf(buffer)
        self.events.append(
            (
                "frame",
                {
                    "pcm": audio,
                    "sample_rate": frame.get_sample_rate(),
                    "bytes_per_sample": frame.get_bytes_per_sample(),
                    "channels": frame.get_number_of_channels(),
                    "samples": frame.get_samples_per_channel(),
                },
            )
        )

    def of(self, name: str) -> list[dict]:
        return [payload for event, payload in self.events if event == name]

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def pcm(self) -> bytes:
        return b"".join(frame["pcm"] for frame in self.of("frame"))

    def ttfb(self) -> list[int]:
        return [
            metric["metrics"]["ttfb"]
            for metric in self.of("metrics")
            if "ttfb" in metric["metrics"]
        ]

    def usage(self) -> list[dict]:
        """The usage reported at the end of each request, then at stop."""
        return [
            metric["metrics"]
            for metric in self.of("metrics")
            if "output_characters" in metric["metrics"]
        ]

    def said(self) -> str:
        """Everything sent and everything logged, as one text to search."""
        logged = [call for log in self._logs for call in log.call_args_list]
        return repr(self.events) + repr(logged)


class Body(httpx.AsyncByteStream):
    """A response body that follows a script.

    `("advance", ms)` moves the clock, `("pcm", bytes)` is a chunk,
    `("cut",)` breaks the connection, `("hold",)` waits to be closed.
    """

    def __init__(self, kalpa: "Kalpa", steps: tuple) -> None:
        self.kalpa = kalpa
        self.steps = steps

    async def __aiter__(self):
        sent = 0
        for step, *args in self.steps:
            # A chunk is on its way downstream before anything else happens,
            # as it is when the next one takes time to arrive.
            await self.kalpa.heard(sent)
            if step == "advance":
                self.kalpa.clock.advance(args[0])
            elif step == "pcm":
                sent += len(args[0])
                yield args[0]
            elif step == "cut":
                raise httpx.ReadError("the connection was cut")
            elif step == "hold":
                self.kalpa.holding.set()
                await asyncio.Future()
        await self.kalpa.heard(sent)

    async def aclose(self) -> None:
        self.kalpa.closed += 1


class Kalpa:
    """Answers each request with the next scripted answer.

    `("audio", *steps)` is a 200 with a `Body`; `("refuse", waited_ms,
    status, error_type)` is Kalpa's error body after a wait; `("raise",
    exception)` is what the HTTP library raises instead of an answer.
    """

    def __init__(
        self, clock: FakeClock, ten_env: RecordingTenEnv, answers: list
    ) -> None:
        self.clock = clock
        self.ten_env = ten_env
        self.answers = answers
        self.texts: list[str] = []
        self.urls: list[str] = []
        self.holding = asyncio.Event()
        # Responses closed, whether read to the end or not.
        self.closed = 0
        # Audio that had gone downstream when the current answer began.
        self._before = 0

    async def heard(self, sent: int) -> None:
        """Wait until the whole samples in `sent` bytes of the current
        answer have gone downstream."""
        expected = self._before + sent - sent % 2
        await until(lambda: len(self.ten_env.pcm()) >= expected)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.texts.append(json.loads(request.content)["text"])
        self.urls.append(str(request.url))
        self._before = len(self.ten_env.pcm())
        kind, *args = self.answers.pop(0)
        if kind == "audio":
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm"},
                stream=Body(self, tuple(args)),
            )
        if kind == "raise":
            raise args[0]
        self.clock.advance(args[0])
        return httpx.Response(
            args[1], json={"error": {"type": args[2], "message": "Why."}}
        )


def text_input(text: str, end: bool, request_id: str = "r1") -> Data:
    data = Data.create("tts_text_input")
    data.set_property_from_json(
        None,
        json.dumps(
            {
                "request_id": request_id,
                "text": text,
                "text_input_end": end,
                "metadata": {"session_id": "session"},
            }
        ),
    )
    return data


def flush() -> Data:
    data = Data.create("tts_flush")
    data.set_property_from_json(None, json.dumps({"flush_id": "flush"}))
    return data


async def until(predicate) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 5)


class Harness:
    """Runs the extension from `on_init` to `on_stop`.

    `answers` are Kalpa's, one per request it gets. `properties` are the
    extension's; without `params` it gets a key and nothing else.
    """

    def __init__(self, *answers, **properties) -> None:
        self.clock = FakeClock()
        properties.setdefault("params", {"api_key": "test-key"})
        self.ten_env = RecordingTenEnv(properties)
        self.kalpa = Kalpa(self.clock, self.ten_env, list(answers))
        self.extension = KalpaHttpTTSExtension("kalpa")
        # What the client asked of the HTTP library.
        self.http_options: dict = {}
        real_client = httpx.AsyncClient

        def scripted_client(**options):
            self.http_options = options
            return real_client(
                transport=httpx.MockTransport(self.kalpa.handle),
                headers=options["headers"],
                timeout=options["timeout"],
            )

        self._patches = [
            patch("ten_ai_base.tts2_http.time", self.clock),
            patch("ten_ai_base.tts2_http.datetime", self.clock),
            patch.object(kalpa_tts.httpx, "AsyncClient", scripted_client),
        ]

    async def __aenter__(self) -> "Harness":
        for item in self._patches:
            item.start()
        await self.extension.on_init(self.ten_env)
        await self.extension.on_start(self.ten_env)
        return self

    async def __aexit__(self, *_exc) -> None:
        try:
            # A stop that raises takes the whole runtime down with it.
            await self.extension.on_stop(self.ten_env)
            await asyncio.gather(
                self.extension.loop_task, return_exceptions=True
            )
        finally:
            for item in self._patches:
                item.stop()

    async def send(self, data: Data) -> None:
        await self.extension.on_data(self.ten_env, data)

    def ends(self) -> list[dict]:
        return self.ten_env.of("tts_audio_end")

    def errors(self) -> list[dict]:
        return self.ten_env.of("error")

    def idle(self) -> bool:
        return (
            self.extension._processing_request_id is None
            and len(self.extension.input_queue) == 0
        )

    async def finished(self, request_id: str = "r1") -> None:
        """Wait for the audio end of a request and for the base to rest."""
        await until(
            lambda: any(e["request_id"] == request_id for e in self.ends())
            and self.idle()
        )


def accounting(end: dict) -> tuple[int, int, int]:
    return (
        end["reason"],
        end["request_total_audio_duration_ms"],
        end["request_event_interval_ms"],
    )
