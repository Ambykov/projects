# STT SDK

Клиентские библиотеки для STT Service — сервиса распознавания речи (WebSocket + REST).

Два SDK:
- **Python** (`stt-client`) — backend: batch-файлы, микрофон сервера, скрипты.
- **TypeScript** (`stt-client` на npm) — браузер: streaming с микрофона (AudioWorklet).

## Что выбрать

| Сценарий | SDK |
|---|---|
| Backend batch, файлы | Python |
| Backend: микрофон сервера | Python (`[mic]` extra) |
| Frontend: браузер, микрофон | TypeScript |
| Frontend: prepared audio | TypeScript (`sendUtterance`) |
| Node.js / backend | Raw WebSocket (пример в `typescript/examples/node/`) |
| >100 req/s | Raw WS + пул |

## Python SDK

```bash
pip install stt-client
# с микрофоном:
pip install 'stt-client[mic]'
```

```python
import asyncio
from stt_client import STTClient, WavAudioSource

async def main():
    client = STTClient(base_url="ws://localhost:8000/ws/stt", token="your-api-key")
    result = await client.stream(WavAudioSource("audio.wav"))
    print(result.text)

asyncio.run(main())
```

Полная документация — `python/README.md`.

## TypeScript SDK

```bash
npm install stt-client
# или из tarball: npm install /path/to/stt-client-0.2.2.tgz
```

```ts
import { Session } from 'stt-client';

const session = new Session({
  baseUrl: 'ws://localhost:8000/ws/stt',
  token: 'your-api-key',
  workletPath: '/resample-worklet.js',  // см. typescript/README.md §workletPath
  onPartial: (text) => console.log('partial:', text),
  onFinal: (r) => console.log('final:', r.text, r.timings),
});

await session.connect();
await session.startUtterance();
// ... пользователь говорит ...
const result = await session.endUtterance();
console.log(result.text);
await session.close();
```

Полная документация — `typescript/README.md`.

## Документация

- `docs/guides/sdk-python.md` — интеграция Python
- `docs/guides/sdk-typescript.md` — интеграция TS
- `docs/api/websocket.md` — контракт WS
- `docs/api/rest.md` — REST batch
- `docs/api/errors.md` — коды ошибок

## Примеры

- `python/examples/stream_wav.py` — транскрипция WAV
- `typescript/examples/web_demo.html` — браузерная демо
- `typescript/examples/node/transcribe.mjs` — Node.js (raw WebSocket)

## Лицензия

MIT — см. `LICENSE`.

## Upstream

- **STT Service** — отдельный репозиторий (gateway + deploy).
- **GigaSTT** — github.com/ekhodzitsky/gigastt (MIT).
- **GigaAM v3** — github.com/salute-developers/GigaAM (commercial use requires separate license).
