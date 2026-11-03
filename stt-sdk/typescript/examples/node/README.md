# Node.js example: transcribe WAV via STT Gateway

TS SDK (`stt-client`) is **browser-only** (uses `WebSocket`, `AudioContext`, `AudioWorklet`).
For Node.js / backend integration — use the raw WebSocket protocol directly (same protocol as SDK).

Spec: `docs/api/websocket.md`.

## Install

    cd sdk/typescript/examples/node
    npm install

## Run

    STT_API_KEY=your-api-key node transcribe.mjs path/to/audio.wav

For remote server (`<your-server>`), open SSH tunnel first:

    ssh -N -L 8100:127.0.0.1:8100 user@<your-server-ip>

## Expected output

    final: Стой.
    timings: {"upstream_connect_ms":4,"upstream_first_byte_ms":110,"upstream_final_ms":228,"audio_bytes_received":32000}
    client_ttfp_ms: 234
    client_ttft_ms: 350
    partials_count: 0

## What it does

1. Parse WAV (PCM16, 16 kHz, mono) → Float32 → PCM16 bytes
2. WS connect → wait for `{"type": "ready"}`
3. Send audio in 100ms chunks → `{"type": "end_of_utterance"}`
4. Receive `partial` / `final` → send `{"type": "stop"}`
5. Print final text + timings

## Error handling

- `{"type": "error", "code": ...}` — handled (throws). Retry-able: `pool_unavailable`, `session_limit`, `upstream_*`.
- No client-side timeout — add `setTimeout` around the promise if needed.
- See `docs/guides/sdk-python.md` §4 for retry patterns (same protocol).

## Limitations

- No retry (add if needed).
- No streaming partial display (partials logged to stderr).
- Audio must be exactly 16 kHz mono — use `ffmpeg -ar 16000 -ac 1 in.mp3 out.wav`.
