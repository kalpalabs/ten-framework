# kalpa_tts_python guarder evidence

PR head: ccbdd4c48b01fe348035d56d8c2a723249bb9f1d
Command (from ai_agents/): task tts-guarder-test EXTENSION=kalpa_tts_python -- -rA
Result: 77 passed, 12 skipped, 5 failed (the 5 failures are upstream tests that fail identically on upstream main).

kalpa_dump_test_connection_status_request_id_1.pcm: raw PCM, 16000 Hz, 16-bit signed little-endian, mono, 4.32 s (138240 bytes), written by test_connection_status (text: "hello world, hello agora, hello shanghai, nice to meet you!").
Play: ffplay -f s16le -ar 16000 -ch_layout mono kalpa_dump_test_connection_status_request_id_1.pcm
