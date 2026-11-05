# stt-client

Python SDK для Ru Voice Assistant STT Gateway.

Тонкая обёртка над WebSocket API с поддержкой:

- Потоковой транскрипции (partial/final)

- Автоматического retry на retryable ошибки

- AudioSource для WAV-файлов и микрофона

## Установка

```
pip install stt-client
```

С поддержкой микрофона:

```
pip install "stt-client[mic]"
```

## Выбор SDK

Какой SDK использовать в зависимости от сценария:

| Сценарий | SDK | Почему |
|---|---|---|
| **Backend: batch, файлы** | **Python SDK** | `WavAudioSource`, retry, простой API |
| **Backend: микрофон сервера** | **Python SDK** | `MicAudioSource` (нужен физический микрофон, редко) |
| **Frontend: браузер, микрофон** | **TypeScript SDK** | `AudioWorklet`, `getUserMedia`, PCM16 48→16 кГц |
| **Backend: Node.js** | **Raw WebSocket** | TS SDK браузерный; см. `sdk/typescript/examples/node/transcribe.mjs` как шаблон |
| **Высокая нагрузка (>100 req/s)** | **Raw WS + пул** | SDK over-engineered, см. `docs/ops/voice-core-scaling.md` |

**Не смешивайте:** Python SDK не работает в браузере (нет WS API + sounddevice), TS SDK не работает в Node.js (нет AudioWorklet).

## Быстрый старт

### Транскрипция WAV-файла

```
import asyncio
from stt_client import STTClient, WavAudioSource

async def main():
    client = STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token="your-api-key",
    )

    audio = WavAudioSource("audio.wav")
    result = await client.stream(audio)

    print(result.text)

asyncio.run(main())
```

### С callback на partial

```
import asyncio
from stt_client import STTClient, WavAudioSource, PartialResult

async def main():
    client = STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token="your-api-key",
    )

    def on_partial(partial: PartialResult):
        print(f"partial: {partial.text}")

    audio = WavAudioSource("audio.wav")
    result = await client.stream(audio, on_partial=on_partial)

    print(f"final: {result.text}")

asyncio.run(main())
```

### Обработка ошибок

```
import asyncio
from stt_client import STTClient, WavAudioSource, STTError

async def main():
 client = STTClient(
 base_url="ws://localhost:8100/ws/stt",
 token="your-api-key",
 )

 try:
 audio = WavAudioSource("audio.wav")
 result = await client.stream(audio)
 print(result.text)
 except STTError as e:
 if e.is_retryable:
 print(f"retryable error: {e.code}")
 else:
 print(f"non-retryable error: {e.code}")

asyncio.run(main())
```

## Перегрузка сервиса (pool_unavailable)

`STTError('pool_unavailable')` означает, что модель перегружена (все 4 слота заняты, см. `docs/ops/voice-core-scaling.md`).

**В Python SDK retry автоматический** (3 попытки, 1s/2s/4s). Если после 3 попыток всё ещё `pool_unavailable` — нагрузка устойчиво выше ёмкости.

**Рекомендуемый паттерн для UI:**

```
from stt_client import STTError

try:
    result = await client.stream(WavAudioSource("audio.wav"))
except STTError as e:
    if e.code == "pool_unavailable":
        # 3 retry не помогли — сигнал эксплуатации масштабировать
        show_user_message("Сервис перегружен, повторите через несколько секунд")
    elif e.code == "unauthorized":
        # 4401 — неверный API_KEY, retry бессмыслен
        log_and_alert("Invalid API key")
    else:
        show_user_message(f"Ошибка: {e.code}")
```

**Команде эксплуатации:** если `pool_unavailable` регулярно (>5% запросов в час) — увеличить pool 4→6→8 (5 минут, throughput ×2). См. `docs/ops/voice-core-scaling.md`.


### Использование из микрофона

```
import asyncio
from stt_client import STTClient, MicAudioSource

async def main():
    client = STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token="your-api-key",
    )

    audio = MicAudioSource()
    result = await client.stream(audio)

    print(result.text)

asyncio.run(main())
```

## API Reference

### STTClient

```
class STTClient:
    def __init__(
        base_url: str,
        token: str,
        retry_policy: RetryPolicy = None,
        timeout_sec: float = 30.0,
    )

    async def stream(
        audio: AudioSource,
        on_partial: Callable[[PartialResult], None] = None,
    ) -> FinalResult
```

### RetryPolicy

```
class RetryPolicy:
    def __init__(
        max_attempts: int = 3,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 10.0,
    )
```

Экспоненциальная задержка: 1s, 2s, 4s, 8s... (max 10s)

### AudioSource

```
class AudioSource:
    @classmethod from_wav(path: str) -> WavAudioSource
    @classmethod from_mic() -> MicAudioSource
    async def stream() -> AsyncIterator[bytes]
```

## Retryable ошибки

Следующие коды ошибок автоматически повторяются (`RetryPolicy`, 3 попытки, 1s/2s/4s):

**In-session errors** (ADR-0014 Gap 1, `{"type": "error"}`):

- `pool_unavailable` — нет свободных слотов модели
- `upstream_connect_failed` — не удалось подключиться к GigaSTT
- `upstream_ready_timeout` — таймаут ожидания ready от GigaSTT
- `upstream_recv_timeout` — таймаут получения данных от GigaSTT

**Close codes** (WebSocket 4401/4408/4429 → `STTError`):

- `session_limit` (4429) — превышен `MAX_CONCURRENT_SESSIONS`, retry с backoff
- `idle_timeout` (4408) — сессия простаивала > `SESSION_IDLE_TIMEOUT_SEC`, переподключиться

**Non-retryable** (не повторяются):

- `unauthorized` (4401) — неверный API_KEY
- `protocol_error`, `client_timeout`, `connection_closed`, `ws_error`

Паритет с TS SDK (`sdk/typescript/README.md` §Ошибки) и с реализацией в `stt_client/errors.py` + `stt_client/client.py`.

## Best practices

Проверенные практикой грабли — из реальной интеграции (demo + review).

### ✅ DO

- **Хранить `API_KEY` в env / secrets manager** (KeePass, Vault). Никогда — в коде, git, чате, логах.
- **Обрабатывать `STTError` по `e.code`** — разные коды требуют разных реакций (см. §«Retryable ошибки»).
- **Логировать `final.timings`** — единственный способ увидеть реальный TTFT и latency.
- **Для фраз >5 сек — сегментировать** (TTFT линейный, ADR-0016). 3–5-сек куски, склеить `final.text`.
- **`on_partial` — замена текста**: `resultText = partial.text`, не `+=`. Partial — текущая гипотеза модели на весь сегмент, не «очередное слово».
- **`asyncio.gather` для нескольких стримов** — SDK поддерживает concurrent (до `MAX_CONCURRENT_SESSIONS=20`).
- **Явно указывать `timeout_sec`** — default 30s покрывает 15-сек фразы, для длинных увеличьте.

### ❌ DON'T

- **Не вызывать `subprocess.run`, `time.sleep`, `requests.get` внутри `async def`** — блокирует event loop. В FastAPI/async-приложении все остальные запросы ждут. Используйте `asyncio.to_thread`, `httpx.AsyncClient`, `asyncio.sleep`. **Реальный кейс:** demo-сервис вызывал `subprocess.run(ffmpeg)` в async — при 5 одновременных запросах все ждали по 100–200 мс.
- **Не накапливайте partial'ы** (`resultText += partial.text`) — получите «наст наст настоящий наст настоящий свод». Partial — замена.
- **Не `asyncio.wait_for(client.stream(...), timeout=5)`** — TTFT на длинных фразах 10+ сек. `timeout_sec` (default 30) уже защищает от зависания.
- **Не ждать partial на фразах <1 сек** — GigaSTT не даёт (ADR-0005). Сразу ждите `final`.
- **Не использовать `MicAudioSource` на сервере** без физического микрофона — только `WavAudioSource`.
- **Не отправлять WAV-заголовок** через WebSocket — принимает **raw PCM16** без header (см. `docs/api/websocket.md` §Формат фреймов).
- **Не отправлять 48 kHz** — SDK ожидает 16 kHz mono. Ресемплируйте заранее: `ffmpeg -ar 16000 -ac 1 in.mp3 out.wav`.
- **Не полагаться на partial'ы для «полного текста»** — они могут переиграть гипотезу (partial 3 короче partial 2). Только `final` — окончательный.

## Требования

- Python >= 3.11

- websockets >= 13.1

- httpx >= 0.27

Для микрофона:

- sounddevice

- numpy


## Ссылки

- **Гайд по интеграции в наш проект:** [`docs/guides/sdk-python.md`](../../docs/guides/sdk-python.md) — сервер `serverllm`, SSH-туннель, обработка перегрузки, FAQ.

- **Пайплайн STT → LLM:** [`docs/guides/voice-integration.md`](../../docs/guides/voice-integration.md).

- **Контракт WS:** `docs/api/websocket.md`.

- **Ошибки:** `docs/api/errors.md`.

- **Масштабирование:** `docs/ops/voice-core-scaling.md`.

- **ADR-0016:** TTFT на длинных фразах.

- **ADR-0017:** архитектура SDK (one connection per utterance).
