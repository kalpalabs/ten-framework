"""Kalpa sendText text and responseAudio audio, one request per connection."""

import base64
import json

from .base import ConnectionObserver


class KalpaObserver(ConnectionObserver):
    def text_context(self, raw, request_id):
        message = json.loads(raw)
        # The session message and the empty keepalive carry no text to speak.
        if message.get("type") == "sendText" and (message.get("text") or "").strip():
            # Requests are serialized, and an interrupted one ends its connection.
            return request_id
        return None

    def audio_context(self, raw):
        message = json.loads(raw)
        audio = message.get("pcm_b64")
        if message.get("type") == "responseAudio" and audio:
            if base64.b64decode(audio, validate=True):
                return self.active_context
        return None
