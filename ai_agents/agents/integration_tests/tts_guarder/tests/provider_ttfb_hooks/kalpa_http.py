"""Observe kalpa HTTP adapter request and audio boundaries."""

from .http import HttpAdapterHook


class Hook(HttpAdapterHook):
    module_name = "ten_packages.extension.kalpa_http_tts.kalpa_tts"
    client_class = "KalpaHttpTTSClient"
