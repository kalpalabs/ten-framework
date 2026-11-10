#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
from dataclasses import dataclass, field
import os
import time
import traceback

from pydantic import ValidationError

from ten_ai_base.const import LOG_CATEGORY_KEY_POINT, LOG_CATEGORY_VENDOR
from ten_ai_base.helper import PCMWriter
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
    ModuleType,
    TTSAudioEndReason,
)
from ten_ai_base.struct import TTSTextInput
from ten_ai_base.tts2 import AsyncTTS2BaseExtension
from ten_runtime import AsyncTenEnv

from .config import DEFAULT_SAMPLE_RATE, KalpaTTSConfig
from .kalpa_tts import KalpaTTSClient, KalpaTTSError

# Kalpa's limit on the text of one utterance. A longer request continues in a
# second utterance.
MAX_UTTERANCE_CHARS = 8000
# Times one request is sent again after its connection dropped before any of
# its audio arrived.
MAX_REPLAYS = 2
# Failed request ids remembered, to recognize their input arriving late.
MAX_ENDED_REQUESTS = 64


def _split(text: str, limit: int) -> list[str]:
    """Cut text longer than one utterance at the last space before the limit."""
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit) + 1 or limit
        pieces.append(text[:cut])
        text = text[cut:]
    pieces.append(text)
    return pieces


@dataclass
class _Request:
    """One TEN request: the text sent for it and the audio received."""

    request_id: str
    # Every message of the request, `(text, flush)`, in order. It is kept so
    # the request can be sent again on a new connection.
    ops: list[tuple[str, bool]] = field(default_factory=list)
    # Messages of `ops` already sent on the current connection.
    sent: int = 0
    # Characters in the utterance that is still open.
    open_chars: int = 0
    # Utterances ended, and how many of them Kalpa has finished speaking.
    flushes: int = 0
    done: int = 0
    input_ended: bool = False
    replays: int = 0
    # Monotonic times of the first text sent and of `tts_audio_start`.
    send_started_at: float | None = None
    audio_started_at: float | None = None
    audio_bytes: int = 0
    # Set once the request has ended. Nothing is emitted for it afterwards,
    # except by the task that ends it.
    closed: bool = False
    end_task: asyncio.Task | None = None


class KalpaTTSExtension(AsyncTTS2BaseExtension):
    """Kalpa text-to-speech over the streaming WebSocket.

    Each `tts_text_input` is forwarded as it arrives. A request is one Kalpa
    utterance, which Kalpa starts speaking at its first complete sentences
    and keeps prosodically continuous; `text_input_end` ends it. The base
    class runs one request at a time, so one utterance is open at a time.

    A flush drops the connection and opens a new one, so neither text nor
    audio of the interrupted request can reach the next one.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config: KalpaTTSConfig | None = None
        self.client: KalpaTTSClient | None = None
        self._request: _Request | None = None
        # Ids of requests that ended in an error, oldest first.
        self._ended: dict[str, None] = {}
        self._recorders: dict[str, PCMWriter] = {}
        # One sender at a time keeps a request's messages in order when a
        # reconnect sends them again while new input is arriving.
        self._send_lock = asyncio.Lock()
        # The task emitting the last events of the request that ended last.
        self._ending: asyncio.Task | None = None
        self._init_error: str | None = None
        self._stopped = False

    # ── Lifecycle ──────────────────────────────────────────────────

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        try:
            await super().on_init(ten_env)
            config_json, _ = await ten_env.get_property_to_json("")
            if not config_json or config_json.strip() == "{}":
                raise ValueError(
                    "Configuration is empty. "
                    "Required parameter 'api_key' is missing."
                )
            config = KalpaTTSConfig.model_validate_json(config_json)
            config.update_params()
            ten_env.log_info(
                f"config: {config.to_str(sensitive_handling=True)}",
                category=LOG_CATEGORY_KEY_POINT,
            )
            config.check()
            self.config = config
            # Connect now, so the session is ready before the first text.
            self._start_client()
        except ValidationError as exc:
            # A validation error quotes the rejected value, which may be the
            # key, so only its locations and messages are reported.
            await self._fail_init(
                "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: "
                    f"{error['msg']}"
                    for error in exc.errors(
                        include_input=False, include_url=False
                    )
                )
            )
        except Exception as exc:
            await self._fail_init(str(exc))

    async def _fail_init(self, message: str) -> None:
        self._init_error = f"Initialization failed: {message}"
        self.ten_env.log_error(f"on_init failed: {message}")
        await self.send_tts_error(
            request_id="",
            error=self._module_error(
                KalpaTTSError("invalid_config", self._init_error), fatal=True
            ),
        )

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        self._stopped = True
        request, self._request = self._request, None
        if request is not None:
            request.closed = True
        client, self.client = self.client, None
        if client is not None:
            client.stop()
        for request_id in list(self._recorders):
            await self._close_recorder(request_id)
        await super().on_stop(ten_env)

    def vendor(self) -> str:
        return "kalpa"

    def vendor_metadata(self) -> dict:
        if self.config is None:
            return {}
        return {
            "url": self.config.websocket_url,
            "model": self.config.model,
            "voice_id": self.config.voice_id,
        }

    def synthesize_audio_sample_rate(self) -> int:
        return self.config.sample_rate if self.config else DEFAULT_SAMPLE_RATE

    def _start_client(self) -> None:
        self.client = KalpaTTSClient(self.config, self.ten_env, self)
        self.client.start()

    # ── Input ──────────────────────────────────────────────────────

    async def request_tts(self, t: TTSTextInput) -> None:
        request_id = t.request_id
        if self._stopped:
            # The base class has dropped its request state; anything emitted
            # now would go out without the request's metadata.
            return
        try:
            if request_id in self._ended:
                # More input for a request that failed before its
                # `text_input_end`. The base class has made it the active
                # request again; release it so the next one can start.
                await self.finish_request(
                    request_id, reason=TTSAudioEndReason.ERROR
                )
                return

            request = self._request
            if request is None or request.request_id != request_id:
                request = self._begin_request(request_id)

            client = self.client
            if client is None:
                await self._end_request(
                    request,
                    TTSAudioEndReason.ERROR,
                    self._module_error(
                        KalpaTTSError(
                            "invalid_config",
                            self._init_error or "The extension is stopped",
                        ),
                        fatal=True,
                    ),
                )
                return
            if client.failed is not None:
                if client.failed.permanent:
                    await self._end_request(
                        request,
                        TTSAudioEndReason.ERROR,
                        self._module_error(client.failed, fatal=True),
                    )
                    return
                # The connection was given up on after repeated failures.
                # A new request is a reason to try again.
                self._start_client()

            self._journal(request, t.text or "")
            if t.text_input_end:
                request.input_ended = True
                if request.open_chars:
                    self._journal_flush(request)
            await self._drain(request)
            await self._complete_if_done(request)
        except Exception:
            self.ten_env.log_error(
                f"request_tts failed for {request_id}: "
                f"{traceback.format_exc()}"
            )
            request = self._request
            if request is not None and request.request_id == request_id:
                await self._fail_request(
                    request,
                    KalpaTTSError("internal_error", "request_tts failed"),
                )

    def _begin_request(self, request_id: str) -> _Request:
        previous = self._request
        if previous is not None and not previous.closed:
            # The base class starts a request only after the previous one
            # ended, so this is not expected.
            self.ten_env.log_warn(
                f"Request {previous.request_id} was still open when "
                f"{request_id} started"
            )
            previous.closed = True
        request = _Request(request_id)
        self._request = request
        # A request that ended on half a sample must not shift this one.
        self.leftover_bytes = b""
        if self.config is not None and self.config.dump:
            os.makedirs(self.config.dump_path, exist_ok=True)
            self._recorders[request_id] = PCMWriter(
                os.path.join(
                    self.config.dump_path, f"kalpa_dump_{request_id}.pcm"
                )
            )
        return request

    def _journal(self, request: _Request, text: str) -> None:
        # Text is sent as it is: Kalpa appends each message to the utterance,
        # and the spaces between sentences are part of the input. Whitespace
        # before the first word would open an utterance with nothing to say.
        if not text or (not request.open_chars and not text.strip()):
            return
        for piece in _split(text, MAX_UTTERANCE_CHARS):
            if request.open_chars + len(piece) > MAX_UTTERANCE_CHARS:
                self._journal_flush(request)
            request.ops.append((piece, False))
            request.open_chars += len(piece)

    @staticmethod
    def _journal_flush(request: _Request) -> None:
        request.ops.append(("", True))
        request.flushes += 1
        request.open_chars = 0

    async def _drain(self, request: _Request) -> None:
        """Send the request's messages that the connection has not had yet."""
        async with self._send_lock:
            while not request.closed and request.sent < len(request.ops):
                client = self.client
                if client is None or not await client.wait_ready():
                    # The client gave up or was stopped; whoever caused that
                    # ends the request.
                    return
                if request.closed or request.sent >= len(request.ops):
                    return
                text, flush = request.ops[request.sent]
                replays = request.replays
                try:
                    sent_at = await client.send_text(text, flush)
                except KalpaTTSError:
                    # The connection dropped. The client reports it, and the
                    # request is sent again or failed there.
                    return
                if request.closed:
                    return
                if request.replays != replays:
                    # The connection dropped while this was being sent, and
                    # the request starts over on the next one.
                    continue
                request.sent += 1
                if text.strip():
                    if request.send_started_at is None:
                        # The connection was ready and the message went out
                        # right after this time: the start of TTFB.
                        request.send_started_at = sent_at
                    self.metrics_add_output_characters(len(text))
                    await self.send_tts_request_metrics(
                        request_id=request.request_id, request_text=text
                    )

    # ── Events from the client ─────────────────────────────────────

    async def on_session_ready(self) -> None:
        request = self._request
        if request is not None and not request.closed:
            await self._drain(request)

    async def on_audio(self, pcm: bytes, received_at: float) -> None:
        request = self._request
        if request is None or request.closed:
            return
        if request.audio_started_at is None:
            request.audio_started_at = time.monotonic()
            await self.send_tts_audio_start(request_id=request.request_id)
            if request.closed:
                return
            if request.send_started_at is not None:
                ttfb_ms = max(
                    0, round((received_at - request.send_started_at) * 1000)
                )
                await self.send_tts_ttfb_metrics(
                    request_id=request.request_id, ttfb_ms=ttfb_ms
                )
                if request.closed:
                    return

        recorder = self._recorders.get(request.request_id)
        if recorder is not None:
            await recorder.write(pcm)
            if request.closed:
                return
        # Counted and sent with nothing in between, so the reported duration
        # is the duration of the audio that went downstream.
        request.audio_bytes += len(pcm)
        self.metrics_add_recv_audio_chunks(pcm)
        await self.send_tts_audio_data(pcm)

    async def on_response_done(self, status: str) -> None:
        request = self._request
        if request is None or request.closed:
            return
        if status != "completed":
            await self._fail_request(
                request,
                KalpaTTSError(
                    "response_cancelled",
                    "Kalpa ended the response before it completed",
                ),
            )
            return
        request.done += 1
        await self._complete_if_done(request)

    async def on_response_error(self, error: KalpaTTSError) -> None:
        self.ten_env.log_error(
            f"vendor_error: {error.error_type}: {error.message}",
            category=LOG_CATEGORY_VENDOR,
        )
        request = self._request
        if request is not None and not request.closed:
            await self._fail_request(request, error)

    async def on_connection_failed(
        self, error: KalpaTTSError, will_retry: bool
    ) -> None:
        request = self._request
        active = request is not None and not request.closed
        if not will_retry:
            module_error = self._module_error(error, fatal=True)
            if active:
                await self._end_request(
                    request, TTSAudioEndReason.ERROR, module_error
                )
            else:
                await self.send_tts_error(request_id="", error=module_error)
            return
        if not active:
            # An idle connection that drops is replaced without an error.
            return

        replayable = (
            request.audio_started_at is None
            and error.response_id is None
            and request.replays < MAX_REPLAYS
        )
        if not replayable:
            # Audio has gone downstream, or Kalpa failed the utterance:
            # sending the text again would repeat or fail the same way.
            await self._end_request(
                request,
                TTSAudioEndReason.ERROR,
                self._module_error(error, fatal=False),
            )
            return
        # Nothing was heard yet: keep the request and send its text again
        # once the new session is ready.
        request.replays += 1
        request.sent = 0
        request.done = 0
        await self.send_tts_error(
            request_id=request.request_id,
            error=self._module_error(error, fatal=False),
        )

    # ── Ending a request ───────────────────────────────────────────

    async def cancel_tts(self) -> None:
        # The base class forgets every request on a flush, so an id seen
        # after it is a new request.
        self._ended.clear()
        request = self._request
        if request is not None:
            # A new session, so text Kalpa has not spoken and audio not yet
            # read are gone before the next request.
            await self._end_request(
                request, TTSAudioEndReason.INTERRUPTED, replace_client=True
            )
        elif self._ending is not None and not self._ending.done():
            # The request was already ending. Its last events go out before
            # the base class reports the flush as done.
            await asyncio.shield(self._ending)

    async def _complete_if_done(self, request: _Request) -> None:
        if (
            request.input_ended
            and not request.closed
            and request.sent == len(request.ops)
            and request.done >= request.flushes
        ):
            await self._end_request(request, TTSAudioEndReason.REQUEST_END)

    async def _fail_request(
        self, request: _Request, error: KalpaTTSError
    ) -> None:
        """End a request that failed while its session is still open.

        Kalpa may still hold text or audio of it, so the session is replaced
        before the next request can use it.
        """
        await self._end_request(
            request,
            TTSAudioEndReason.ERROR,
            self._module_error(error, fatal=False),
            replace_client=True,
        )

    async def _end_request(
        self,
        request: _Request,
        reason: TTSAudioEndReason,
        error: ModuleError | None = None,
        replace_client: bool = False,
    ) -> None:
        """End the request once. The events survive a cancelled caller."""
        if request.end_task is None:
            # Everything up to the task runs without a pause: from here on
            # nothing else is emitted for the request, and a client that is
            # replaced can no longer deliver anything.
            request.closed = True
            if self._request is request:
                self._request = None
            if reason == TTSAudioEndReason.ERROR:
                # Its remaining input may still arrive.
                self._remember_ended(request.request_id)
            client = None
            if replace_client:
                client, self.client = self.client, None
                if client is not None:
                    client.stop()
            request.end_task = asyncio.create_task(
                self._emit_end(request, reason, error, client is not None)
            )
            self._ending = request.end_task
        await asyncio.shield(request.end_task)

    async def _emit_end(
        self,
        request: _Request,
        reason: TTSAudioEndReason,
        error: ModuleError | None,
        client_stopped: bool,
    ) -> None:
        request_id = request.request_id
        if self._stopped:
            return
        try:
            if client_stopped:
                await self.on_disconnected(
                    code=0,
                    message=(
                        "flush"
                        if reason == TTSAudioEndReason.INTERRUPTED
                        else "request failed"
                    ),
                )
                if not self._stopped:
                    # Started before the base class is told the request is
                    # over, so the next request waits for this session.
                    self._start_client()
            if error is not None:
                await self.send_tts_error(request_id=request_id, error=error)
            await self._close_recorder(request_id)
            # A request that failed before any audio has no audio to end.
            if (
                reason != TTSAudioEndReason.ERROR
                or request.audio_started_at is not None
            ):
                await self.send_tts_audio_end(
                    request_id=request_id,
                    request_event_interval_ms=self._interval_ms(request),
                    request_total_audio_duration_ms=self._duration_ms(request),
                    reason=reason,
                )
                await self.send_usage_metrics(request_id)
            # After a flush the base class clears the request itself.
            if reason != TTSAudioEndReason.INTERRUPTED:
                await self.finish_request(request_id, reason=reason)
        except Exception:
            self.ten_env.log_error(
                f"Ending request {request_id} failed: {traceback.format_exc()}"
            )

    def _remember_ended(self, request_id: str) -> None:
        if len(self._ended) >= MAX_ENDED_REQUESTS:
            self._ended.pop(next(iter(self._ended)))
        self._ended[request_id] = None

    async def _close_recorder(self, request_id: str) -> None:
        recorder = self._recorders.pop(request_id, None)
        if recorder is None:
            return
        try:
            await recorder.flush()
        except Exception as exc:
            self.ten_env.log_error(
                f"Flushing the PCM dump of {request_id} failed: {exc}"
            )

    @staticmethod
    def _interval_ms(request: _Request) -> int:
        """From `tts_audio_start` to now; 0 when no audio started."""
        if request.audio_started_at is None:
            return 0
        return int((time.monotonic() - request.audio_started_at) * 1000)

    def _duration_ms(self, request: _Request) -> int:
        bytes_per_second = (
            self.synthesize_audio_sample_rate()
            * self.synthesize_audio_sample_width()
            * self.synthesize_audio_channels()
        )
        return int(request.audio_bytes * 1000 / bytes_per_second)

    def _module_error(self, error: KalpaTTSError, fatal: bool) -> ModuleError:
        return ModuleError(
            message=error.message,
            module=ModuleType.TTS,
            code=(
                ModuleErrorCode.FATAL_ERROR
                if fatal
                else ModuleErrorCode.NON_FATAL_ERROR
            ),
            vendor_info=ModuleErrorVendorInfo(
                vendor=self.vendor(),
                code=error.error_type,
                message=error.message,
            ),
        )
