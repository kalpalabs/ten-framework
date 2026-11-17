# kalpa_tts_python guarder evidence

PR head: 5c079434f5f1c8b04f5f67ca05b8ee3792f2a001
Command (from ai_agents/): task tts-guarder-test EXTENSION=kalpa_tts_python -- -rA
Result: 75 passed, 12 skipped, 5 failed (the 5 failures are upstream tests that fail identically on upstream main 0c2dc90e8).

kalpa_dump_test_connection_status_request_id_1.pcm: raw PCM, 16000 Hz, 16-bit signed little-endian, mono, 4.24 s (135680 bytes), written by test_connection_status (text: "hello world, hello agora, hello shanghai, nice to meet you!").
Play: ffplay -f s16le -ar 16000 -ch_layout mono kalpa_dump_test_connection_status_request_id_1.pcm
