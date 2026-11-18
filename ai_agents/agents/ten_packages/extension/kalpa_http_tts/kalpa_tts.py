#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""Client for Kalpa's streaming text-to-speech over HTTP.

Protocol: `POST /v1/tts/{voice_id}/stream`
(https://docs.kalpalabs.ai/reference#post-v1-tts-voice-id-stream). One
request speaks one text as one utterance. The response body is the audio
itself, raw 16-bit little-endian mono PCM, and the response starts with the
first audio: a request that fails before then is answered with a status
code and a JSON error, and one that fails later is cut off before its body
completes. Closing the connection stops the generation.

The client speaks HTTP/1.1, where closing a response closes its
connection. Over HTTP/2 the HTTP library would leave the stream open:
Kalpa would generate the rest, and the audio nobody reads would use up the
connection's receive window.
"""
import asyncio
import json
from typing import Any, AsyncIterator, Tuple

import httpx

from ten_ai_base.const import LOG_CATEGORY_VENDOR
from ten_ai_base.struct import TTS2HttpResponseEventType
from ten_ai_base.tts2_http import AsyncTTS2HttpClient
from ten_runtime import AsyncTenEnv

from .config import KalpaHttpTTSConfig

CONNECT_TIMEOUT_S = 10.0
# Deadline for the first audio of a request, and for each chunk after it.
# A request has no total deadline: its audio is as long as its text.
READ_TIMEOUT_S = 20.0
# A connection whose response was read to its end is kept this long, so the
# next request does not wait for a new one. Kalpa keeps an idle connection
# open for longer.
KEEPALIVE_EXPIRY_S = 300.0
# Kalpa's limit on the text of one request. A longer text is spoken in
# several requests.
MAX_TEXT_CHARS = 8000
# Times one text is sent again after Kalpa refused it as overloaded, or the
# connection failed, before any of its audio arrived.
MAX_RESENDS = 2
# Kalpa refuses a request while it is at capacity. Nothing was generated, so
# the text can be sent again: after `Retry-After` (about a second), then
# twice as long.
OVERLOADED = "overloaded"
OVERLOADED_RETRY_DELAY_S = 1.0
RESEND_BASE_DELAY_S = 0.5
RESEND_MAX_DELAY_S = 8.0

PCM_CONTENT_TYPE = "audio/pcm"
BYTES_PER_SAMPLE = 2
# Chunks read ahead of the extension before the response has to wait.
QUEUED_CHUNKS = 32

# Failures no later request can fix: the key, the account, the voice, the
# model or another setting has to change first. The extension checks the
# text itself, so a request Kalpa calls invalid has an invalid setting.
PERMANENT_ERROR_TYPES = frozenset(
    {
        "authentication_error",
        "account_suspended",
        "allowance_exhausted",
        "not_found",
        "invalid_request",
    }
)
# An answer without Kalpa's error body, as from a proxy in front of it, is
# named by its status (`http_403`). These statuses do not change by asking
# again either.
PERMANENT_STATUSES = frozenset({401, 402, 403, 404})


class KalpaHttpTTSError(Exception):
    """A failure reported by Kalpa or by the connection to it."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        permanent: bool = False,
        status: int = 0,
        retry_after: float = 0.0,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        # True when sending again cannot succeed without a configuration
        # change.
        self.permanent = permanent
        # HTTP status, when Kalpa answered with one.
        self.status = status
        # Seconds Kalpa asked to wait before the text is sent again.
        self.retry_after = retry_after
        # Kalpa's id for the request, to quote to its support.
        self.request_id = request_id

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message}"


def _seconds(value: str | None) -> float:
    try:
        seconds = float(value or 0)
    except ValueError:
        return 0.0
    return seconds if 0 < seconds < float("inf") else 0.0


async def _error_from_response(response: httpx.Response) -> KalpaHttpTTSError:
    """Read Kalpa's `{"error": {"type", "message"}}` body of a refusal."""
    status = response.status_code
    error: Any = None
    try:
        body = json.loads(await response.aread())
        error = body.get("error") if isinstance(body, dict) else None
    except ValueError:
        pass
    if not isinstance(error, dict):
        error = {}
    kalpa_type = str(error.get("type") or "")
    return KalpaHttpTTSError(
        kalpa_type or f"http_{status}",
        str(error.get("message") or f"Kalpa answered with HTTP {status}"),
        permanent=(
            kalpa_type in PERMANENT_ERROR_TYPES
            if kalpa_type
            else status in PERMANENT_STATUSES
        ),
        status=status,
        retry_after=_seconds(response.headers.get("retry-after")),
        request_id=str(
            error.get("request_id")
            or response.headers.get("x-request-id")
            or ""
        ),
    )


def _lost_connection(exc: Exception) -> bool:
    """Whether the network, or Kalpa's end of the connection, failed."""
    return isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError))


def _error_from_transport(exc: httpx.TransportError) -> KalpaHttpTTSError:
    if isinstance(exc, httpx.TimeoutException):
        return KalpaHttpTTSError("timeout", "Kalpa did not answer in time")
    if _lost_connection(exc):
        return KalpaHttpTTSError(
            "connection_error", str(exc) or type(exc).__name__
        )
    # The request could not be put on the wire. The HTTP library's text for
    # that can quote a header, and with it the key, so only its kind is
    # reported.
    return KalpaHttpTTSError(
        "request_error",
        f"The request could not be sent ({type(exc).__name__})",
    )


def _speakable(text: str) -> bool:
    """Whether the text has a letter or a digit to say.

    Kalpa rejects blank text, and it answers a lone punctuation mark or
    emoji with a sound.
    """
    return any(char.isalnum() for char in text)


def _split(text: str, limit: int) -> list[str]:
    """Cut text longer than one request at the last space before the limit."""
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit) + 1 or limit
        pieces.append(text[:cut])
        text = text[cut:]
    pieces.append(text)
    return pieces


class _Stream:
    """The audio of one `get()`: fetched by `task`, read from `queue`."""

    def __init__(self) -> None:
        # Whole samples in order, then how the audio ended: None, or the
        # error.
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUED_CHUNKS)
        self.task: asyncio.Task | None = None
        self.cancelled = False


class KalpaHttpTTSClient(AsyncTTS2HttpClient):
    """Speaks one text per request and yields its audio as it arrives."""

    def __init__(
        self,
        config: KalpaHttpTTSConfig,
        ten_env: AsyncTenEnv,
        refusal: str = "",
    ) -> None:
        """`refusal` says why the configuration could not be read, if so."""
        super().__init__()
        self._config = config
        self._ten_env = ten_env
        if not refusal:
            try:
                config.validate()
            except ValueError as exc:
                refusal = str(exc)
        # Why no request can be made, if so. Each one is then answered
        # with this, and nothing is sent.
        self._refusal = f"Initialization failed: {refusal}" if refusal else ""
        self._http = httpx.AsyncClient(
            # Never log these headers: they carry the API key.
            headers=(
                {} if refusal else {"Authorization": f"Bearer {config.api_key}"}
            ),
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_S),
        )
        self._stream: _Stream | None = None

    async def get(
        self, text: str, request_id: str
    ) -> AsyncIterator[Tuple[bytes | None, TTS2HttpResponseEventType]]:
        """Yield the audio of `text` as whole samples, then how it ended."""
        if self._refusal:
            yield (
                self._refusal.encode("utf-8"),
                TTS2HttpResponseEventType.INVALID_KEY_ERROR,
            )
            return
        if not _speakable(text):
            yield None, TTS2HttpResponseEventType.END
            return

        stream = _Stream()
        stream.task = asyncio.create_task(self._fetch(text, request_id, stream))
        self._stream = stream
        try:
            while True:
                item = await stream.queue.get()
                if stream.cancelled:
                    # Audio that was read but not yet handed on is dropped.
                    yield None, TTS2HttpResponseEventType.FLUSH
                    return
                if isinstance(item, bytes):
                    yield item, TTS2HttpResponseEventType.RESPONSE
                    continue
                if item is None:
                    yield None, TTS2HttpResponseEventType.END
                elif item.permanent:
                    yield (
                        str(item).encode("utf-8"),
                        TTS2HttpResponseEventType.INVALID_KEY_ERROR,
                    )
                else:
                    yield (
                        str(item).encode("utf-8"),
                        TTS2HttpResponseEventType.ERROR,
                    )
                return
        finally:
            if self._stream is stream:
                self._stream = None
            if not stream.cancelled:
                # The caller stopped reading before the audio ended.
                stream.cancelled = True
                stream.task.cancel()

    async def cancel(self) -> None:
        """Stop the request in flight. `get()` yields no more of its audio."""
        stream = self._stream
        if stream is None or stream.task is None or stream.cancelled:
            return
        stream.cancelled = True
        # Closes the response where it stands, also before its first byte,
        # and with it the connection: that stops the generation at Kalpa.
        # Once only: a second cancellation would interrupt the closing.
        stream.task.cancel()
        try:
            # Wakes a `get()` that is waiting for audio.
            stream.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def clean(self) -> None:
        await self.cancel()
        await self._http.aclose()

    def get_extra_metadata(self) -> dict[str, Any]:
        return {}

    async def _fetch(self, text: str, request_id: str, stream: _Stream) -> None:
        """Put the audio of `text` on the stream, then how it ended."""
        failure: KalpaHttpTTSError | None = None
        try:
            for piece in _split(text, MAX_TEXT_CHARS):
                if _speakable(piece):
                    await self._speak(piece, request_id, stream)
        except KalpaHttpTTSError as exc:
            failure = exc
        except Exception as exc:
            # Its text is not reported: it may quote the request.
            failure = KalpaHttpTTSError(
                "internal_error", f"Unexpected {type(exc).__name__}"
            )
        if failure is not None:
            self._ten_env.log_error(
                f"vendor_error: {failure} (status={failure.status}, "
                f"kalpa_request_id={failure.request_id}) "
                f"of request_id: {request_id}",
                category=LOG_CATEGORY_VENDOR,
            )
        await stream.queue.put(failure)

    def _unquoted(self, failure: KalpaHttpTTSError) -> KalpaHttpTTSError:
        """No message carries the key, whoever worded it."""
        key = self._config.api_key
        if key:
            failure.message = failure.message.replace(key, "<api_key>")
        return failure

    async def _speak(self, text: str, request_id: str, stream: _Stream) -> None:
        """One request to Kalpa, sent again while that is safe and useful."""
        resends = 0
        while True:
            # Whether this attempt has handed any audio on.
            heard = False
            try:
                async with self._http.stream(
                    "POST",
                    self._config.endpoint,
                    json=self._config.request_body(text),
                ) as response:
                    if response.status_code != 200:
                        raise await _error_from_response(response)
                    content_type = response.headers.get("content-type", "")
                    if not content_type.lower().startswith(PCM_CONTENT_TYPE):
                        raise KalpaHttpTTSError(
                            "unexpected_response",
                            "Kalpa answered with content type "
                            f"'{content_type}', not {PCM_CONTENT_TYPE}",
                        )
                    # A chunk may end in the middle of a sample.
                    leftover = b""
                    async for chunk in response.aiter_bytes():
                        data = leftover + chunk
                        whole = len(data) - len(data) % BYTES_PER_SAMPLE
                        leftover = data[whole:]
                        if whole:
                            heard = True
                            await stream.queue.put(data[:whole])
                return
            except KalpaHttpTTSError as exc:
                failure = self._unquoted(exc)
                if exc.error_type != OVERLOADED or resends == MAX_RESENDS:
                    raise
                delay = exc.retry_after or OVERLOADED_RETRY_DELAY_S
            except httpx.TransportError as exc:
                failure = self._unquoted(_error_from_transport(exc))
                if (
                    # A timeout would only be waited out again, and a
                    # request that cannot be sent fails the same way.
                    not _lost_connection(exc)
                    # Audio has gone downstream: sending the text again
                    # would repeat speech.
                    or heard
                    or resends == MAX_RESENDS
                ):
                    raise failure from None
                delay = RESEND_BASE_DELAY_S
            delay = min(delay * 2**resends, RESEND_MAX_DELAY_S)
            resends += 1
            self._ten_env.log_warn(
                f"vendor_status: {failure}; sending the text again in "
                f"{delay:.1f}s ({resends}/{MAX_RESENDS}) "
                f"of request_id: {request_id}",
                category=LOG_CATEGORY_VENDOR,
            )
            await asyncio.sleep(delay)
