# Python SDK — руководство по интеграции

**Пакет:** `stt-client` v0.1.1

**Полная документация пакета:** [`sdk/python/README.md`](../../sdk/python/README.md) (установка, API reference, примеры, best practices).

**Этот гайд** — про интеграцию в **наш проект** (сервер `serverllm`, API-ключи voice-core, обработка перегрузки).


## 1. Куда подключаться

### Локальная разработка

Gateway запущен на `localhost:8100`:

```
from stt_client import STTClient

client = STTClient(
    base_url="ws://localhost:8100/ws/stt",
    token="testtoken",
)
```

### Сервер (`serverllm`)

Прод-сервер **не отдаёт STT-порт наружу** (вариант A — только внутренние приложения). Для отладки с ноутбука:

```
# SSH-туннель: локальный 8100 → серверный gateway (127.0.0.1:8100)
ssh -p 2222 -N -L 8100:127.0.0.1:8100 llmuser@192.168.0.45
```

После этого SDK работает с тем же `ws://localhost:8100/ws/stt`, но запросы идут на сервер. **Маркер успеха:** `client.stream(...)` вернёт `timings`, а `curl http://localhost:8100/version` покажет `"public_hostname": "serverllm"` (не `localhost`).

### Внутри `infra_public_net` (для контейнеров)

Если ваше приложение — Docker-контейнер в `infra_public_net` (как `litellm-proxy`, `normik-stand`):

```
base_url = "ws://stt-gateway:8000/ws/stt"   # имя сервиса, не localhost
```

**Обязательно:** добавьте `stt-gateway` и IP хоста в `no_proxy` (см. `09.03`), иначе запросы уйдут через корпоративный прокси и получат `403`.


## 2. API-ключ

`API_KEY` выдаёт администратор voice-core. Формат: 64 hex-символа (генерируется `openssl rand -hex 32`).

**Хранение:** KeePass / Bitwarden / Vault. **Не в git, не в env-переменных с логированием, не в чатах.**

```
import os
client = STTClient(
    base_url=os.environ["STT_URL"],
    token=os.environ["STT_API_KEY"],
)
```

**В Docker-контейнере** — через `.env` + `environment:` в compose, не через `docker run -e` (попадает в `docker inspect`).

**Ротация:** ежеквартально (см. `11.04.01`). При получении нового ключа — передеплой приложения с новым env.


## 3. Быстрый старт

```
import asyncio
from stt_client import STTClient, WavAudioSource

async def transcribe(path: str) -> str:
    client = STTClient(
        base_url="ws://localhost:8100/ws/stt",
        token="your-api-key",
        timeout_sec=30.0,        # client-side timeout на final
    )
    result = await client.stream(WavAudioSource(path))
    return result.text

print(asyncio.run(transcribe("audio.wav")))
```

Аудио: **WAV, PCM16, mono, 16 kHz**. Если у вас 48 kHz — ресемплируйте (`ffmpeg -ar 16000 -ac 1`) до отправки.


## 4. Обработка `pool_unavailable` (важно)

Streaming API (`client.stream()`) захватывает аудио и отправляет его. При перегрузке модели (`pool_unavailable`) **аудио уже потеряно** — retry внутри SDK переотправит его, но caller должен решить, показать ли это пользователю как ошибку.

**Автоматический retry есть** (3 попытки, 1s/2s/4s). Если после 3 попыток `pool_unavailable` — значит, нагрузка устойчиво выше ёмкости сервиса.

**Рекомендуемый паттерн для UI:**

```
from stt_client import STTClient, WavAudioSource, STTError

try:
    result = await client.stream(WavAudioSource("audio.wav"))
except STTError as e:
    if e.code == "pool_unavailable":
        # 3 retry не помогли — это сигнал эксплуатации масштабировать сервис
        # (см. docs/ops/voice-core-scaling.md: pool 4→6→8→12→16)
        show_user_message("Сервис перегружен, повторите через несколько секунд")
    elif e.code == "unauthorized":
        # 4401 — неверный API_KEY, retry бессмыслен
        log_and_alert("Invalid API key")
    else:
        # прочие — общая ошибка
        show_user_message(f"Ошибка: {e.code}")
```

**Что делать команде эксплуатации:** если `pool_unavailable` приходит регулярно (>5% запросов в час), см. `docs/ops/voice-core-scaling.md` — увеличить pool (4→6→8), это 5 минут, throughput ×2.

**Полный whitelist retryable-кодов** (6 штук: 4 in-session + 2 close code 4429/4408) — в `sdk/python/README.md` §«Retryable ошибки». SDK ретраит их автоматически через `RetryPolicy` (3 попытки, 1s/2s/4s). `unauthorized` (4401) — не ретраится, сразу алерт админу.

## 5. Обработка `client_timeout`

`client_timeout` означает, что final не пришёл за `timeout_sec` (default 30s). Причины:

- Модель зависла (`gigastt` не отвечает).

- Очень длинная фраза (>30s аудио — TTFT > 30s, ADR-0016).

**TTFT растёт линейно с длиной фразы** (offline-модель). Для длинных фраз (>5s) либо:

- Увеличить `timeout_sec` (например, 60s для фраз 30s).

- Разбить аудио на сегменты по 3–5s (вручную или через VAD — planned в 4.C).

- Использовать buffered API с ретраями.


## 6. Примеры

### REST batch (один файл)

См. `sdk/python/examples/stream_wav.py` — рабочий пример.

```
STT_API_KEY=your-key python sdk/python/examples/stream_wav.py audio.wav
```

### asyncio-приложение (несколько файлов)

```
import asyncio
from stt_client import STTClient, WavAudioSource

async def main(files: list[str]):
    client = STTClient(base_url="...", token="...")
    results = await asyncio.gather(
        *[client.stream(WavAudioSource(f)) for f in files],
        return_exceptions=True,
    )
    for f, r in zip(files, results):
        if isinstance(r, Exception):
            print(f"{f}: error {r}")
        else:
            print(f"{f}: {r.text}")
```

**Ограничение:** gateway принимает до `MAX_CONCURRENT_SESSIONS=20` одновременных WS. Если файлов больше — `asyncio.Semaphore(20)`.

### Интеграция с LLM (STT → LLM)

См. `docs/guides/voice-integration.md` — типичный пайплайн с `litellm-proxy`.


## 7. Тесты SDK

```
# Юнит (без gateway, mock):
pytest sdk/python -v
# → 8 passed, 1 skipped

# Интеграционные (нужен работающий gateway на localhost:8100):
export STT_API_KEY=$(cat ~/.secrets/voice-core-api-key.txt)
pytest sdk/python -v --run-integration
# → 9 passed
```

Если `STT_API_KEY` не задан — integration-тест берёт `testtoken` (default, работает только на локальном gateway с AUTH_MODE=none).


## 8. Частые проблемы

| Симптом | Причина | Решение |
| - | - | - |
| `STTError('unauthorized')` (4401) | Неверный API_KEY | Проверить `echo ${#STT_API_KEY}` = 64 |
| `STTError('pool_unavailable')` | Модель перегружена (4 слота) | Retry; при регулярных — pool 4→6 (scaling.md) |
| `STTError('client_timeout')` | Фраза >30s, или модель зависла | Увеличить `timeout_sec`; VAD для длинных фраз |
| `ConnectionRefusedError` | Gateway не запущен / неверный URL | `curl http://localhost:8100/health` |
| `ModuleNotFoundError: stt_client` | SDK не установлен | `pip install -e sdk/python` |
| `ValueError: expected 16000 Hz` | Аудио не 16 kHz | `ffmpeg -ar 16000 -ac 1 in.mp3 out.wav` |
| Тихо возвращает `""` | Аудио — тишина / очень короткая фраза (<1s) | GigaSTT плохо распознаёт <1s (ADR-0015) |
| `final` пришёл, но текст **обрезан** на фразе 10–15 сек | `upstream_final = 15s` хардкод в gateway (см. `websocket.md` §Тайминги) | Сегментировать на 3–5-сек куски; VAD planned (ADR-0022) |
| `on_partial` не вызывается на коротких фразах (<2 сек) | GigaSTT не даёт partial на коротких сегментах | Норма, ADR-0005. Ждите `final` |
| UI показывает **«наст наст настоящий наст настоящий свод»** | `resultText += partial.text` — накопление вместо замены | `resultText = partial.text` (замена); см. `websocket.md` §partial |
| `stream()` зависает >30s без ответа | `timeout_sec=30` default, но final не приходит | Увеличить `timeout_sec`; проверить gateway `curl /health`; логи `docker logs voice-core-stt-gateway` |
| Весь сервис тормозит при 3+ одновременных запросах | `subprocess.run` / `time.sleep` в `async def` — блокирует event loop | `await asyncio.to_thread(...)` для sync-операций (ffmpeg, file I/O) |


## 8.5. Do's and Don'ts (кратко)

Полный список — в `sdk/python/README.md` §«Best practices». Здесь — самые частые грабли.

### ✅ DO

- **`API_KEY` — в env / secrets manager**, не в коде, не в git, не в логах.
- **`on_partial` — замена** (`resultText = partial.text`), не накопление (`+=`).
- **Для фраз >5 сек — сегментировать** на 3–5-сек куски (ADR-0016), или ждать VAD.
- **Логировать `final.timings`** — единственный способ увидеть реальный TTFT.
- **`asyncio.gather` для concurrent** — SDK поддерживает, до 20 сессий.
- **`await asyncio.to_thread(func, ...)`** для sync I/O внутри `async def` (ffmpeg, `subprocess`, чтение больших файлов).
- **`timeout_sec` явно** — default 30s покрывает 15-сек фразы; для 30-сек фраз — увеличьте до 60.

### ❌ DON'T

- **Не `resultText += partial.text`** — partial это замена, получите дубли.
- **Не `subprocess.run` / `time.sleep` / `requests.get` в `async def`** — блокирует event loop.
- **Не `asyncio.wait_for(client.stream(), timeout=5)`** — TTFT на длинных фразах 10+ сек, `timeout_sec` уже покрывает.
- **Не ждать partial на <1 сек** — GigaSTT не даёт (ADR-0005).
- **Не отправлять WAV-заголовок** через WS — только raw PCM16.
- **Не отправлять 48 kHz** — только 16 kHz mono, ресемплируйте заранее.
- **Не `MicAudioSource` на сервере** без физического микрофона.


## 9. Ссылки

- `sdk/python/README.md` — API reference, установка, примеры кода, best practices.

- `docs/api/websocket.md` — протокол WS (ready/partial/final/error/close codes), TTFT таблица, sequence diagram.

- `docs/api/errors.md` — все коды ошибок и retry-able.

- `docs/ops/voice-core-scaling.md` — план масштабирования (если `pool_unavailable`).

- ADR-0016 — TTFT на длинных фразах.

- ADR-0017 — архитектура SDK (one connection per utterance).
