#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""Runs the extension in the TEN test runtime and records what it emits."""
import asyncio
import json
from typing import Any, Awaitable, Callable

from ten_runtime import (
    AsyncExtensionTester,
    AsyncTenEnvTester,
    AudioFrame,
    Data,
    TenError,
    TenErrorCode,
)

EXTENSION = "kalpa_tts_python"
END_REASON_REQUEST_END = 1
END_REASON_INTERRUPTED = 2
END_REASON_ERROR = 3
FATAL_ERROR = -1000
NON_FATAL_ERROR = 1000


class Recording(AsyncExtensionTester):
    """Runs `script`, then stops; or stops as soon as `done(self)` is true."""

    def __init__(
        self,
        script: Callable[["Recording", AsyncTenEnvTester], Awaitable[None]],
        done: Callable[["Recording"], bool] | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        super().__init__()
        self._script = script
        self._done = done
        self._timeout_s = timeout_s
        self._tasks: list[asyncio.Task] = []
        # Everything received, in order: `(name, payload)`. Audio frames
        # appear as `("pcm_frame", {"bytes": n, "sample_rate": hz})`.
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.audio = bytearray()

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def of(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]

    def index(self, name: str) -> int:
        return self.names().index(name)

    def ended(self, request_id: str) -> bool:
        return any(
            end["request_id"] == request_id for end in self.of("tts_audio_end")
        )

    def errors(self) -> list[dict[str, Any]]:
        return self.of("error")

    def ttfb(self) -> list[int]:
        return [
            metric["metrics"]["ttfb"]
            for metric in self.of("metrics")
            if "ttfb" in metric.get("metrics", {})
        ]

    async def send_text(
        self,
        ten_env: AsyncTenEnvTester,
        request_id: str,
        text: str,
        end: bool = False,
        turn_id: int = 1,
    ) -> None:
        data = Data.create("tts_text_input")
        data.set_property_from_json(
            "",
            json.dumps(
                {
                    "request_id": request_id,
                    "text": text,
                    "text_input_end": end,
                    "metadata": {"session_id": "session", "turn_id": turn_id},
                }
            ),
        )
        await ten_env.send_data(data)

    async def send_flush(self, ten_env: AsyncTenEnvTester) -> None:
        data = Data.create("tts_flush")
        data.set_property_from_json(
            "",
            json.dumps(
                {"flush_id": "flush", "metadata": {"session_id": "session"}}
            ),
        )
        await ten_env.send_data(data)

    async def wait_for(
        self, condition: Callable[["Recording"], bool], timeout_s: float = 10.0
    ) -> None:
        """Wait inside a script until the recording satisfies `condition`."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while not condition(self):
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("condition not met")
            await asyncio.sleep(0.01)

    async def on_start(self, ten_env: AsyncTenEnvTester) -> None:
        async def run_script() -> None:
            try:
                await self._script(self, ten_env)
            except Exception as exc:
                ten_env.stop_test(
                    TenError.create(
                        TenErrorCode.ErrorCodeGeneric,
                        f"script failed: {type(exc).__name__}: {exc}",
                    )
                )
                return
            if self._done is None:
                ten_env.stop_test()

        async def watchdog() -> None:
            await asyncio.sleep(self._timeout_s)
            ten_env.stop_test(
                TenError.create(
                    TenErrorCode.ErrorCodeGeneric,
                    f"timed out; received {self.names()}",
                )
            )

        self._tasks = [
            asyncio.create_task(run_script()),
            asyncio.create_task(watchdog()),
        ]

    async def on_stop(self, ten_env: AsyncTenEnvTester) -> None:
        for task in self._tasks:
            task.cancel()

    def _check(self, ten_env: AsyncTenEnvTester) -> None:
        if self._done is not None and self._done(self):
            ten_env.stop_test()

    async def on_data(self, ten_env: AsyncTenEnvTester, data: Data) -> None:
        payload, _ = data.get_property_to_json("")
        self.events.append((data.get_name(), json.loads(payload or "{}")))
        self._check(ten_env)

    async def on_audio_frame(
        self, ten_env: AsyncTenEnvTester, audio_frame: AudioFrame
    ) -> None:
        buffer = audio_frame.lock_buf()
        try:
            pcm = bytes(buffer)
        finally:
            audio_frame.unlock_buf(buffer)
        self.audio += pcm
        self.events.append(
            (
                "pcm_frame",
                {
                    "bytes": len(pcm),
                    "sample_rate": audio_frame.get_sample_rate(),
                },
            )
        )
        self._check(ten_env)


def run(
    config: dict[str, Any],
    script: Callable[[Recording, AsyncTenEnvTester], Awaitable[None]],
    done: Callable[[Recording], bool] | None = None,
    timeout_s: float = 15.0,
) -> Recording:
    """Run the extension with `config` and return what it emitted."""
    recording = Recording(script, done, timeout_s)
    recording.set_test_mode_single(EXTENSION, json.dumps(config))
    error = recording.run()
    assert error is None, error.error_message()
    return recording
