# kalpa_http_tts

Text-to-speech from [Kalpa](https://kalpalabs.ai) for TEN voice agents, over
Kalpa's streaming HTTP endpoint
([`POST /v1/tts/{voice_id}/stream`](https://docs.kalpalabs.ai/reference#post-v1-tts-voice-id-stream)).
The default model, `kalpa-tts-1`, speaks English, Hindi and code-switched
Hinglish.

The extension speaks each piece of text it receives on its own (see
[One request per text](#one-request-per-text)). Kalpa also has a streaming
WebSocket endpoint, which takes a whole reply as one utterance and is the
better fit for an agent that sends a reply in pieces; the `kalpa_tts_python`
extension uses it and takes the same settings. Use this extension where a
WebSocket is not an option.

## Configuration

Create an API key at <https://kalpalabs.ai/dashboard/keys/> and pass it in
`params`. Only `api_key` is required.

```json
{
  "params": {
    "api_key": "${env:KALPA_TTS_API_KEY}",
    "voice_id": "18c26f49-9f9c-400c-bec9-7467f28e1db5",
    "model": "kalpa-tts-1",
    "sample_rate": 24000
  }
}
```

| Property | Default | Meaning |
|---|---|---|
| `params.api_key` | required | Kalpa API key, without spaces or line breaks. It is sent as the `Authorization` header only. |
| `params.voice_id` | Arjun | A voice id from `GET /v1/voices`. A voice is offered for some models only; `GET /v1/voices?model=kalpa-tts-1` lists that model's voices. |
| `params.model` | `kalpa-tts-1` | A model id from `GET /v1/models`. |
| `params.sample_rate` | `24000` | Output rate in Hz: `8000`, `16000`, `22050`, `24000`, `32000`, `44100` or `48000`. Kalpa generates at 24 kHz and resamples to the others on its side. Audio is 16-bit mono PCM. Any other value fails at start. |
| `params.base_url` | `https://api.kalpalabs.ai` | Host of another Kalpa deployment. |
| `params.url` | unset | Full HTTP endpoint, used as given; `{voice_id}` in it is replaced by the voice. Wins over a top-level `url`, which wins over `params.base_url`. |
| `dump`, `dump_path` | `false`, `/tmp` | Write each request's audio to `kalpa_dump_<request_id>.pcm`. |

Every other key in `params` is passed to Kalpa as a generation parameter:
`temperature`, `acoustic_temperature` and `audio_quality` are described in
the [Kalpa documentation](https://docs.kalpalabs.ai/text-to-speech#generation-parameters).
`params.chunk_length_schedule`, a setting of the WebSocket endpoint, is
accepted and not used.

The endpoint is logged, and reported in each event's metadata, as scheme,
host and path only: user info and query parameters may hold a key or a
token.

## Behavior

### One request per text

Each `tts_text_input` is one request to Kalpa and one utterance, sent when
it arrives, and its audio is forwarded while Kalpa generates it. This is
the contract of TEN's HTTP base class, which the extension uses as it is,
and the endpoint's: one text per request.

An agent that cuts a reply after each punctuation mark therefore gets
each clause as a separate utterance, with its own intonation, where the
WebSocket endpoint is given the whole reply. The extension does not
join pieces itself: it is not told which text is the last of a request, so
text it held back for a longer utterance could be left unspoken, and
holding text back would delay the first audio.

- A text with no letter or digit (whitespace, a lone punctuation mark, an
  emoji) is not sent: there is nothing to say, and Kalpa answers such a
  text with a sound. TEN's voice assistant example drops such pieces by
  the same rule.
- A text longer than Kalpa's 8,000 characters is spoken in several
  requests.
- The requests of one reply reuse one connection.
- Requests are spoken one after another. A request that arrives while
  another is open waits until that one has ended.

### Interruption

`tts_flush` closes the response in flight at once, also one that has not
sent its first byte. The extension speaks HTTP/1.1, where that closes the
connection, and Kalpa stops generating when its connection closes. The
request after a flush opens a new connection. Text waiting behind the
interrupted response is dropped, and no audio of the interrupted request
is forwarded afterwards. If the request had started audio it gets
`tts_audio_end` with reason `INTERRUPTED`; if it had not, there is no
audio to end.

### Audio accounting

- `request_total_audio_duration_ms` is the audio that was forwarded, added
  up over the texts of the request. Audio is forwarded in whole samples: a
  chunk that ends in the middle of a sample is completed by the next one,
  and half a sample at the very end of a response is neither forwarded nor
  counted.
- `request_event_interval_ms` runs from `tts_audio_start` to
  `tts_audio_end`. It is `0` for a request that ends without audio.
- A request gets one `tts_audio_end`, also when a flush arrives while it is
  ending.
- **TTFB** runs from the request for the first text of a `request_id` to
  its first audio. It includes setting up the connection when a new one is
  needed, and the wait before a text is sent again.

### Errors

- **Fatal** (`FATAL_ERROR`): a missing, malformed or rejected key, a
  suspended account, a used-up allowance, an unknown voice or model, and
  invalid settings. The extension does not send such a text again. After
  settings that fail at start, every request is answered with the same
  error and nothing is sent to Kalpa.
- An answer without Kalpa's error body, as from a proxy in front of it,
  is named by its status, for example `http_403`; 401, 402, 403 and 404
  are fatal.
- **Overloaded.** When Kalpa refuses a text as `overloaded` (HTTP 503: it
  is at capacity and generated nothing), the extension sends it again
  after `Retry-After` (about one second), then after twice as long, as
  Kalpa's documentation asks. A request whose connection fails before any
  of its audio arrived is sent again the same way, after half a second.
  A text is sent at most three times; the caller sees only the audio, or a
  `NON_FATAL_ERROR` when the last attempt fails too.
- **Not sent again:** a response that is cut off after audio started,
  since the text would be spoken twice; a failed generation; and a request
  Kalpa does not answer within 20 seconds, since a second wait only adds
  silence. These are `NON_FATAL_ERROR`.
- A `NON_FATAL_ERROR` for a text that is not the last of its request does
  not end the request: the rest of the reply is still spoken. One for the
  last text ends it with `tts_audio_end` and reason `ERROR`.
- The key is never logged, and no error quotes it: an error of the HTTP
  library about a request it could not send is reported by its kind only.

### Subtitles

Kalpa does not return word timestamps, so the extension sends no
`tts_text_result`.

## Tests

The standalone tests run the extension against a local server that
answers like Kalpa's endpoint (`tests/mock_server.py`), and check the audio
accounting on a clock they control (`tests/test_audio_accounting.py`).
They need no key.

```bash
task test-extension EXTENSION=agents/ten_packages/extension/kalpa_http_tts
```

The guarder runs against the Kalpa API with `KALPA_TTS_API_KEY` set:

```bash
task tts-guarder-test EXTENSION=kalpa_http_tts CONFIG_DIR=tests/configs
```
