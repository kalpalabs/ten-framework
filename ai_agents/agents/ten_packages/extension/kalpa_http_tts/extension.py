#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
from datetime import datetime
import os

from pydantic import ValidationError

from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    TTSAudioEndReason,
)
from ten_ai_base.tts2_http import (
    AsyncTTS2HttpClient,
    AsyncTTS2HttpConfig,
    AsyncTTS2HttpExtension,
)
from ten_runtime import AsyncTenEnv

from .config import DEFAULT_SAMPLE_RATE, SAMPLE_RATES, KalpaHttpTTSConfig
from .kalpa_tts import KalpaHttpTTSClient

NON_FATAL_ERROR = int(ModuleErrorCode.NON_FATAL_ERROR.value)


class KalpaHttpTTSExtension(AsyncTTS2HttpExtension):
    """Kalpa text-to-speech over the streaming HTTP endpoint.

    The base class sends each `tts_text_input` to the client as it arrives
    and owns the request's events. Each text is one request to Kalpa and
    one utterance, so a reply that arrives in pieces is spoken piece by
    piece. Kalpa's WebSocket endpoint takes a whole reply as one utterance.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config: KalpaHttpTTSConfig | None = None
        self.client: KalpaHttpTTSClient | None = None
        # Why the settings could not be read, if so.
        self._config_error = ""
        # The request whose `tts_audio_end` has gone out: its id and the
        # time the base class stamped it with when it started, which tells
        # it from a later request with the same id.
        self._ended: tuple[str, datetime | None] | None = None

    async def create_config(self, config_json_str: str) -> AsyncTTS2HttpConfig:
        try:
            return KalpaHttpTTSConfig.model_validate_json(config_json_str)
        except ValidationError as exc:
            # A validation error quotes the rejected value, which may be the
            # key, so only its locations and messages are reported.
            self._config_error = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: "
                f"{error['msg']}"
                for error in exc.errors(include_input=False, include_url=False)
            )
            raise ValueError(self._config_error) from None

    async def create_client(
        self, config: AsyncTTS2HttpConfig, ten_env: AsyncTenEnv
    ) -> AsyncTTS2HttpClient:
        if config is None:
            # No settings could be read. The client answers every request
            # with the failure reported at start-up, and sends nothing.
            return KalpaHttpTTSClient(
                KalpaHttpTTSConfig(),
                ten_env,
                refusal=self._config_error or "Configuration is empty.",
            )
        if config.dump:
            os.makedirs(config.dump_path, exist_ok=True)
        return KalpaHttpTTSClient(config, ten_env)

    def vendor(self) -> str:
        return "kalpa"

    def vendor_metadata(self) -> dict:
        if self.config is None:
            return {}
        return {
            "url": self.config.safe_url,
            "model": self.config.model,
            "voice_id": self.config.voice_id,
        }

    def synthesize_audio_sample_rate(self) -> int:
        # The base class divides by this rate, also after a start-up that
        # rejected the configured one.
        rate = self.config.sample_rate if self.config else DEFAULT_SAMPLE_RATE
        return rate if rate in SAMPLE_RATES else DEFAULT_SAMPLE_RATE

    # The three methods below adjust what the base class does at the end of
    # a request. Each calls the base method for everything else.

    def _calculate_request_event_interval_ms(self) -> int:
        # The interval runs from `tts_audio_start` to `tts_audio_end`. For
        # a request without audio the base class counts from the request;
        # that interval is 0.
        if self.request_ts is None:
            return 0
        return super()._calculate_request_event_interval_ms()

    async def _send_audio_end_and_finish(
        self,
        request_id: str,
        reason: TTSAudioEndReason,
        log_message: str | None = None,
    ) -> None:
        # A flush can meet a request that is ending, or end one whose last
        # text is being spoken. The base class then sends both the
        # interruption and the request's own end; only the first is sent,
        # and it finishes the request.
        ended = (request_id, self.sent_ts)
        if (
            self._ended is not None
            and self._ended[0] == request_id
            and self._ended[1] is self.sent_ts
        ):
            return
        self._ended = ended
        await super()._send_audio_end_and_finish(
            request_id, reason, log_message
        )

    async def _handle_error_with_text_input_end(
        self,
        request_id: str,
        error: ModuleError,
        text_input_end: bool,
    ) -> None:
        if not text_input_end and error.code == NON_FATAL_ERROR:
            # One text of the request failed in a way the next may not, so
            # the request stays open for the rest of its text. The base
            # class finishes it here and still speaks that text: without a
            # `tts_audio_end`, and holding back the request after it.
            await self.send_tts_error(request_id=request_id, error=error)
            return
        await super()._handle_error_with_text_input_end(
            request_id, error, text_input_end
        )
