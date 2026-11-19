# kalpa_tts_python guarder evidence

PR head: 667285ef1a03611d0f63d38be2b454a2a55d3e3e
Command (from ai_agents/): task tts-guarder-test EXTENSION=kalpa_tts_python -- -rA
Result: 77 passed, 12 skipped, 5 failed (the failures are upstream tests that fail identically on upstream main).

kalpa_dump_test_connection_status_request_id_1.pcm: raw PCM, 16000 Hz, 16-bit signed little-endian, mono, 5.12 s (163840 bytes), written by test_connection_status (text: "hello world, hello agora, hello shanghai, nice to meet you!").
Play: ffplay -f s16le -ar 16000 -ch_layout mono kalpa_dump_test_connection_status_request_id_1.pcm
