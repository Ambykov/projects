# WebSocket API

## Endpoint

```
ws://<host>:8100/ws/stt?token=<API_KEY>
```

Альтернативная аутентификация через заголовок:

```
Authorization: Bearer <API_KEY>
```

## Формат фреймов

| Тип | Описание |
| - | - |
| Binary | PCM16 @ 16 kHz, mono, **без WAV заголовка** |


**Важно:** WebSocket принимает сырой PCM16 без WAV заголовка. Отправка WAV файла приведёт к ошибке расшифровки.

## Протокол

### Client → Server

1. **Аудио фреймы** (binary PCM16):

```
<binary PCM16 audio data @ 16kHz>
```

2. **Конец реплики**:

```
{"type": "end_of_utterance"}
```

3. **Конец сессии**:

```
{"type": "stop"}
```

### Server → Client

1. **Готовность** (сразу после подключения):

```
{"type": "ready", "model": "gigaam-v3-rnnt", "sample_rate": 16000, "version": "2.22.0"}
```

2. **Частичный результат** (streaming):

```
{"type": "partial", "text": "..."}
```

> \[!warning\] `partial` — это **замена**, а не накопление

> `partial.text` содержит **текущую гипотезу модели на весь сегмент**, а не «очередное слово». При каждом новом partial **заменяйте** отображение, а не добавляйте к предыдущему.

> ✅ **Правильно** (замена):

```
onPartial(text) { resultEl.textContent = text; }
```

❌ **Неправильно** (накопление — получите «наст наст настоящий наст настоящий свод»):

```
onPartial(text) { resultEl.textContent += text; }
```

Окончательный текст — в `final`. Partial'ы — только для «живого» UI.

3. **Финальный результат**:

```
{"type": "final", "text": "...", "timings": {
    "upstream_connect_ms": 4,
    "upstream_first_byte_ms": 963,
    "upstream_final_ms": 10423,
    "audio_bytes_received": 327680
}}
```

*Примечание:* `timings` — опциональное поле, добавлено в Stage 4.B. Для совместимости проверяйте наличие перед использованием.

## Тайминги

`final` содержит поле `timings` с серверными метриками:

- `upstream_connect_ms` — время подключения к GigaSTT (мс)

- `upstream_first_byte_ms` — время до первого partial (TTFP сервера, мс)

- `upstream_final_ms` — время до final (TTFT сервера, мс)

- `audio_bytes_received` — сколько байт аудио получено от клиента

**TTFP зависит от длины фразы:**

- Короткие (\<1.5 сек): 100–300 мс

- Средние (1.5–3 сек): 300–800 мс

- Длинные (\>3 сек): 800–2000 мс (GigaSTT ждёт контекста)

**TTFT (время до final) линейно растёт с длиной фразы** — это ограничение offline-модели GigaAM v3, не SDK. Модель обрабатывает **весь буфер** после `end_of_utterance`, и время инференса ≈ длина фразы + константа (ADR-0016).

| Длина фразы | TTFT сервер | Рекомендация клиенту |
| - | - | - |
| 1 сек | ~250 мс | ок, показывать final как есть |
| 3 сек | ~3 сек | показывать partial'ы в UI (живой текст) |
| 5 сек | ~5 сек | на грани UX — partial'ы + индикатор |
| 10 сек | ~10 сек | сегментировать на 3–5-сек куски (клиент) |
| 15+ сек | ~15+ сек | **обрезается** gateway'ем по `upstream_final = 15s` (хардкод, ADR-0021 TODO) |


**Что делать:** для реплик \>5 сек — сегментируйте аудио на клиенте по паузам (VAD, planned в SDK, см. ADR-0022) или по фиксированным 3–5-сек кускам. Каждый кусок → отдельный `stream()` → склеить `final.text`.

`client_timeout` (30 сек в Python SDK) — это **не** TTFT, это защита от зависания. Для фраз \>10 сек — увеличить `timeout_sec`.

Клиентский TTFP (`performance.now()` от «Говорить» до первого partial) может отличаться от серверного (`upstream_first_byte_ms`), потому что включает сеть и ресемплинг в браузере (~500–700 мс разница).

## Жизненный цикл сессии

1. Client подключается → Gateway отправляет `ready`

2. Client отправляет аудио → Gateway открывает upstream к GigaSTT

3. Client отправляет `end_of_utterance` → Gateway отправляет `stop` в upstream, закрывает upstream после `final`

4. Дождитесь `final`, затем повторите шаги 2-3 для следующей реплики (overlapping не поддерживается, см. ниже)

5. Client отправляет `stop` → Gateway закрывает сессию

```
sequenceDiagram
    participant C as Client
    participant G as Gateway
    participant U as GigaSTT (upstream)

    C->>G: connect ?token=...
    G->>C: {"type":"ready", ...}
    loop Каждая реплика (per-utterance upstream)
        C->>G: audio chunks (binary PCM16)
        G->>U: open WS + configure + audio
        U-->>G: partial
        G-->>C: {"type":"partial", text}
        U-->>G: partial
        G-->>C: {"type":"partial", text}
        C->>G: {"type":"end_of_utterance"}
        G->>U: stop
        U-->>G: final
        G-->>C: {"type":"final", text, timings}
        Note over G,U: upstream закрывается после final
    end
    C->>G: {"type":"stop"}
    G->>C: close 1000
```

**Ключевое:**

- Клиентское WS-соединение — **long-lived** (на всю сессию диалога).

- Upstream к GigaSTT — **per-utterance** (новое соединение на каждую реплику; GigaAM v3 не держит long-lived WS, ADR-0006).

- `partial` — только **между** `audio` и `final` (не после `final`).

- `final` — после `end_of_utterance`; ждать `final` перед следующей репликой (Gap 2).

## Overlapping utterances

**Текущее поведение (Stage 4.B):** НЕ поддерживается.

Клиент **должен** дождаться `final` перед отправкой аудио следующей реплики. Если отправить `bytes` до получения `final`, аудио **потеряется** (Gap 2 в `gateway-as-is.md`).

**Рекомендация для SDK:** блокировать кнопку «Говорить» до получения `final`, или показывать «ассистент думает».

**Barge-in:** не поддерживается. Для barge-in требуется to-be архитектура (отмена `current_rep_task` при новом `bytes`).

## Ошибки

**`{"type": "error", "code": ..., "message": ...}` отправляется клиенту при ошибках upstream.**

| Code | Описание | Частота | Действие клиента |
| - | - | - | - |
| `upstream_ready_timeout` | GigaSTT не ответил на ready за 10s (нет свободного слота) | **основной** при нагрузке | Retry 2–5 сек |
| `upstream_recv_timeout` | GigaSTT не ответил за 30s (зависание) | редкий | Retry 2–5 сек, если повторяется — сообщить |
| `upstream_connect_failed` | Не удалось подключиться к GigaSTT | редкий | Retry 5–10 сек с backoff |
| `pool_unavailable` | `/ready` показал `pool_available=0` в момент проверки | **очень редкий** (race: только если все слоты заняты ровно в момент проверки) | Retry 1–2 сек; в реальном диалоге чаще приходит `upstream_ready_timeout` |


**Для SDK:** `pool_unavailable` и `upstream_ready_timeout` — **оба означают «нет ёмкости»**, обрабатывайте одинаково (retry с backoff). `pool_unavailable` — защитный код, срабатывает редко, в основном на пиковой нагрузке при синтетических сценариях. В реальном диалоге под нагрузкой ждите `upstream_ready_timeout`.

**Сессия НЕ закрывается** после `error` (кроме `idle_timeout`, `session_limit`, `auth`). Клиент может отправить новую реплику.

**Пример:**

```
{"type": "error", "code": "upstream_ready_timeout", "message": "GigaSTT did not respond to ready in 10s"}
```

**Gap 1 закрыт** (ADR-0014 to-be). Ранее `{"type": "error"}` не отправлялся — клиент видел только close codes.

## Коды закрытия

| Код | Описание |
| - | - |
| 4401 | Unauthorized (неверный или отсутствующий токен) |
| 4408 | Idle timeout (сессия простаивала \> SESSION\_IDLE\_TIMEOUT\_SEC) |
| 4429 | Session limit exceeded (достигнут MAX\_CONCURRENT\_SESSIONS) |
| 1000 | Normal close (клиент отправил `stop`) |


## Пример (Python)

```
import asyncio
import json
import wave
import websockets

async def main():
    uri = "ws://localhost:8100/ws/stt?token=testtoken"
    async with websockets.connect(uri) as ws:
        # Ожидать ready
        ready = await ws.recv()
        print(f"Connected: {ready}")
        
        # Отправить аудио: WAV -> сырой PCM16
        with wave.open("audio.wav", "rb") as wav:
            # Проверка: 16 кГц, mono
            assert wav.getframerate() == 16000, f"Expected 16kHz, got {wav.getframerate()}"
            assert wav.getnchannels() == 1, f"Expected mono, got {wav.getnchannels()}"
            
            # Читаем чанками по 3200 байт (100 мс @ 16kHz, 16-bit)
            while True:
                chunk = wav.readframes(1600)  # 1600 frames * 2 bytes = 3200 байт
                if not chunk:
                    break
                await ws.send(chunk)
        
        # Конец реплики
        await ws.send(json.dumps({"type": "end_of_utterance"}))
        
        # Получить результат
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"{data['type']}: {data.get('text', '')}")
            if data['type'] == 'final':
                break
        
        # Закрыть сессию
        await ws.send(json.dumps({"type": "stop"}))

asyncio.run(main())
```

