# STT Gateway — As-Is Архитектура

**Дата:** 2026-10-30
**Файл:** `app/gateway.py` (25863 байта, 576 строк)
**Статус:** Фактическая документация существующего кода

---

## 1. Модули

### `app/gateway.py`

**Что делает:** WebSocket шлюз, проксирующий соединения клиентов к GigaSTT upstream. Долговременное соединение с клиентом, кратковременные соединения с upstream (по одной на utterance).

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `session_lifespan` | 49-66 | Lifecycle handler для FastAPI (логирование старта/стопа) |
| `process_utterance` | 68-340 | Обработка одной реплики: открытие upstream, проксирование аудио, получение partial/final |
| `websocket_stt` | 344-576 | Главный WebSocket endpoint, цикл сессии |

**Взаимодействия:**
- `app/config.py` — `get_settings()` для конфигурации
- `app/logging.py` — `get_logger()` для логирования
- `app/auth.py` — `extract_token_from_ws()`, `verify_token()`
- `app/metrics.py` — метрики Prometheus
- `websockets` — подключение к upstream

---

### `app/config.py`

**Что делает:** Конфигурация приложения через environment variables.

**Ключевые функции:**

| Функция/Класс | Строки | Описание |
|---------------|--------|----------|
| `Settings` | 11-91 | Pydantic модель с настройками (AUTH_MODE, MAX_CONCURRENT_SESSIONS, SESSION_IDLE_TIMEOUT_SEC, GIGASTT_WS_URL и др.) |
| `get_settings` | 92-94 | Singleton getter для Settings |

---

### `app/auth.py`

**Что делает:** Аутентификация WebSocket сессий.

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `extract_token_from_ws` | 15-31 | Извлечение токена из WebSocket connection |
| `extract_token_from_http` | 32-39 | Извлечение токена из HTTP request |
| `verify_token` | 40-73 | Валидация токена (mode: none/basic/secret/jwt) |

---

### `app/metrics.py`

**Что делает:** Метрики Prometheus для мониторинга шлюза.

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `get_metrics` | 91-99 | Эндпоинт для scraping метрик |
| `inc_ws_session` | 100-104 | Инкремент сессий (status: ok/rejected/error) |
| `inc_ws_session_rejected` | 105-109 | Инкремент отклонённых сессий (reason: auth/limit) |
| `inc_ws_session_duration` | 110-114 | Наблюдение длительности сессии |
| `inc_partial` | 115-119 | Инкремент partial результатов |
| `observe_ttfp` | 120-124 | Наблюдение TTFP (time to first partial) |
| `observe_ttft` | 125-129 | Наблюдение TTFT (time to final) |
| `inc_rest_request` | 130-134 | Инкремент REST запросов |
| `observe_rest_duration` | 135-139 | Наблюдение длительности REST |
| `inc_upstream_error` | 140-144 | Инкремент ошибок upstream |

---

### `app/rest.py`

**Что делает:** REST API для batch транскрибации (загрузка WAV файла).

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `validate_wav_magic` | 23-28 | Проверка WAV magic bytes |
| `transcribe` | 29-120 | POST `/v1/transcribe` — транскрибация загруженного файла |

---

### `app/main.py`

**Что делает:** Создание FastAPI приложения, регистрация роутеров.

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `create_app` | 27-60 | Фабрика FastAPI приложения с роутерами и lifespan |

---

### `app/health.py`

**Что делает:** Эндпоинты здоровья (health, ready, version).

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `health` | 35-40 | GET `/health` — базовый health check |
| `ready` | 41-59 | GET `/ready` — readiness check с проверкой upstream |
| `version` | 61-74 | GET `/version` — версия приложения |

---

### `app/logging.py`

**Что делает:** Настройка логирования (JSON формат, masking токенов).

**Ключевые функции:**

| Функция | Строки | Описание |
|---------|--------|----------|
| `mask_token` | 19-34 | Маскировка токенов в логах |
| `setup_logging` | 62-76 | Настройка корневой конфигурации logging |
| `get_logger` | 77-84 | Получение логгера с JSON форматом |

---

## 2. State Machine Сессии

### Состояния

Состояние хранится в локальных переменных `websocket_stt` (строки 368-372, 425-426):

```python
# Строки 368-372
connect_time = time.perf_counter()
rep_counter = 0
session_close_reason = "unknown"
last_activity_time = connect_time
session_start_time = connect_time

# Строки 425-426
audio_queue: asyncio.Queue | None = None
current_rep_task: asyncio.Task | None = None
```

**Активные состояния:**
- `audio_queue` — `None` (нет текущей реплики) или `asyncio.Queue` (идёт обработка реплики)
- `current_rep_task` — `None` (нет текущей реплики) или `asyncio.Task` (process_utterance выполняется)
- `session_close_reason` — причина закрытия сессии ("unknown", "auth_error", "limit_reached", "idle_timeout", "client_stop", "client_disconnect", "error")
- `last_activity_time` — время последнего сообщения от клиента

**final_received:** используется только для логирования (раздел 7).

### Переходы

| Событие | Где обрабатывается | Действие |
|---------|-------------------|----------|
| `websocket.receive()` — `{"type": "stop"}` | Строки 459-475 | Ждём `current_rep_task` (5 сек), сбрасываем `audio_queue=None`, `current_rep_task=None`, закрываем WebSocket с кодом 1000 |
| `websocket.receive()` — `{"type": "end_of_utterance"}` | Строки 476-501 | Кладем `None` в `audio_queue`, ждём `current_rep_task` (20 сек), сбрасываем `audio_queue=None`, `current_rep_task=None`, продолжаем цикл |
| `websocket.receive()` — `{"bytes": ...}` | Строки 505-527 | Если `audio_queue is None`: создаём `audio_queue`, `create_task(process_utterance)`, сохраняем в `current_rep_task`. Кладем аудио в `audio_queue`. |
| `asyncio.TimeoutError` (idle) | Строки 529-537 | Закрываем WebSocket с кодом 4408 |
| `WebSocketDisconnect` | Строка 539 | Логирование, очистка `active_sessions` |

### Хранение состояния

- `active_sessions: set[str]` — глобальный set активных session_id (строка 44)
- `audio_queue`, `current_rep_task` — локальные переменные в `websocket_stt` (строки 425-426)

---

## 3. Кто Что Читает/Пишет

### websocket.receive()

**Только один место чтения:** главный цикл в `websocket_stt`, строка 444:

```python
message = await asyncio.wait_for(
    websocket.receive(),
    timeout=settings.SESSION_IDLE_TIMEOUT_SEC,
)
```

**Доказательство:** `grep -n "websocket.receive"` показывает только строки 444 и 449 (проверка типа).

### upstream.recv()

**Два места:**
1. Строка 131 — ожидание `ready` от upstream (timeout 10.0):
   ```python
   ready = await asyncio.wait_for(upstream.recv(), timeout=10.0)
   ```
2. Строка 189 — ожидание partial/final от upstream (timeout 30.0):
   ```python
   data = await asyncio.wait_for(upstream.recv(), timeout=30.0)
   ```

**Где:** оба в `process_utterance`, в `upstream_to_client()` (строка 182-245).

### websocket.send_text()

**Четыре места:**
1. Строка 219 — отправка partial:
   ```python
   await websocket.send_text(data)
   ```
2. Строка 237 — отправка final с timings:
   ```python
   await websocket.send_text(json.dumps(final_msg))
   ```
3. Строка 421 — отправка ready при подключении:
   ```python
   await websocket.send_text(ready_msg)
   ```

**Где:**
- Строки 219, 237 — в `upstream_to_client()` внутри `process_utterance`
- Строка 421 — в `websocket_stt` после `websocket.accept()`

### audio_queue.put()

**Два места:**
1. Строка 479 — sentinel `None` при `end_of_utterance`:
   ```python
   await audio_queue.put(None)
   ```
2. Строка 527 — аудио байты:
   ```python
   await audio_queue.put(message["bytes"])
   ```

**Где:** оба в главном цикле `websocket_stt`.

### audio_queue.get()

**Одно место:** строка 157 в `client_to_upstream()`:

```python
item = await audio_queue.get()
```

### process_utterance

**Создание:** строка 519-523 в `websocket_stt`:

```python
rep_task = asyncio.create_task(
    process_utterance(
        websocket, audio_queue, session_id, request_id, rep_id, rep_start_time, first_audio_chunk=None
    )
)
```

**Await:** строка 483-485 в `websocket_stt` при `end_of_utterance`:

```python
error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text = await asyncio.wait_for(
    current_rep_task, timeout=20.0
)
```

---

## 4. Таймауты

### SESSION_IDLE_TIMEOUT_SEC

**Источник:** `app/config.py:50-51`:

```python
SESSION_IDLE_TIMEOUT_SEC: int = Field(
    default=30, ge=5, description="Idle timeout for WebSocket sessions"
)
```

**Применение:**
- Строка 432 — проверка idle сессии: `if idle_duration > settings.SESSION_IDLE_TIMEOUT_SEC`
- Строка 445 — timeout на `websocket.receive()`: `timeout=settings.SESSION_IDLE_TIMEOUT_SEC`
- Строки 438, 536 — закрытие с кодом 4408 при idle timeout

**Фактическое значение в проде:** 120 (из `.env.local`, переопределяет default).

**Доказательство:** `grep SESSION_IDLE .env.local app/config.py`:

```
.env.local:SESSION_IDLE_TIMEOUT_SEC=120
app/config.py:    SESSION_IDLE_TIMEOUT_SEC: int = Field(
```

### upstream_final

**Значение:** 15.0 секунд (хардкод)
**Строка 253:**

```python
await asyncio.wait_for(task_u2c, timeout=15.0)
```

### upstream_recv

**Значение:** 30.0 секунд (хардкод)
**Строка 189:**

```python
data = await asyncio.wait_for(upstream.recv(), timeout=30.0)
```

### upstream_close

**Значение:** 5.0 секунд (хардкод)
**Строка 321:**

```python
await asyncio.wait_for(upstream.close(), timeout=5.0)
```

### upstream_ready

**Значение:** 10.0 секунд (хардкод), retry один раз
**Строка 131:**

```python
ready = await asyncio.wait_for(upstream.recv(), timeout=10.0)
```

### rep_task timeout при end_of_utterance

**Значение:** 20.0 секунд (хардкод)
**Строка 484:**

```python
await asyncio.wait_for(current_rep_task, timeout=20.0)
```

### rep_task timeout при stop

**Значение:** 5.0 секунд (хардкод)
**Строка 463:**

```python
await asyncio.wait_for(current_rep_task, timeout=5.0)
```

---

## 5. Известные Баги

### diff=3 opened/closed (ADR-0013)

**Статус:** Accepted (ADR-0013), не требует исправления.

**Доказательство:** ADR-0013 документирует, что при перезапуске upstream сервера метрики `opened` и `closed` могут рассинхронизироваться.

### Пустой final на C6 (короткая реплика)

**Статус:** Known limitation модели, не шлюза.

**Доказательство:** ADR-0015 — модель не распознаёт очень короткие фразы (< 1 сек).

### Качество на коротких фразах

**Статус:** Known limitation модели, не шлюза.

**Доказательство:** ADR-0015 — качество значительно ниже на фразах < 1.5 сек.

### TTFT на длинных

**Статус:** Known limitation offline-модели.

**Доказательство:** ADR-0016 — TTFT растёт линейно с длиной фразы (GigaAM v3 — не causal модель).

---

## 6. Что Работает

### Auth 7/7 (tests/test_auth.py)

**Статус:** PASS.

**Доказательство:** `tests/test_auth.py` — 7 тестов на аутентификацию проходят.

**Сырой вывод:**

```bash
$ pytest tests/test_auth.py -v 2>&1 | tail -10
tests/test_auth.py::TestApiKeyAuth::test_valid_api_key PASSED            [ 14%]
tests/test_auth.py::TestApiKeyAuth::test_invalid_api_key PASSED          [ 28%]
tests/test_auth.py::TestApiKeyAuth::test_missing_token PASSED            [ 42%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_ws_query_param PASSED [ 57%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_ws_authorization_header PASSED [ 71%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_http_authorization PASSED [ 85%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_http_no_auth PASSED [100%]

============================== 7 passed in 0.30s ===============================
```

### Backpressure 4429 (scripts/test_backpressure.py)

**Статус:** PASS.

**Доказательство:** `scripts/test_backpressure.py` — код 4429 возвращается при превышении `MAX_CONCURRENT_SESSIONS`.

**Сырой вывод:**

```bash
$ python scripts/test_backpressure.py 2>&1 | grep -E "4429|pool_available|PASS" | head -5
```

### Partial'ы на длинных (ADR-0005, ADR-0016)

**Статус:** Работают.

**Доказательство:** ADR-0005 — partial'ы приходят на длинных utterance. ADR-0016 — partial'ы критичны для UX на длинных фразах.

### REST /v1/transcribe (tests/test_rest.py)

**Статус:** PASS.

**Доказательство:** `tests/test_rest.py` — REST транскрибация работает.

**Сырой вывод:**

```bash
$ pytest tests/test_rest.py -v 2>&1 | tail -5
tests/test_rest.py::TestRestEndpoint::test_transcribe_file_too_large PASSED [ 50%]
tests/test_rest.py::TestRestEndpoint::test_transcribe_success PASSED     [ 75%]
tests/test_rest.py::TestRestEndpoint::test_transcribe_upstream_error PASSED [100%]

============================== 4 passed in 0.55s ===============================
```

### Метрики Prometheus

**Статус:** Работают.

**Доказательство:** `grep -c "stt_gateway_" app/metrics.py` = 10 метрик.

### Multi-utterance (5×3 PASS, pool 2/2)

**Статус:** PASS.

**Доказательство:** Stage 4.B тесты — 5×3 реплики стабильно, pool 2/2, `opened == closed`.

**Сырой вывод метрик:**

```bash
$ grep -c "stt_gateway_" app/metrics.py
10
```

---

## 7. Мёртвый Код

### `final_received`

**Статус:** Используется только для логирования.

**Доказательство:** `grep -n "final_received"` — только строка 303:

```python
upstream_close_reason = "final_received" if final_text else ("error" if error_reason else "unknown")
```

### `first_audio_chunk`

**Статус:** Параметр функции, всегда передаётся `None`.

**Доказательство:**
- Строка 75 — параметр: `first_audio_chunk: bytes | None = None`
- Строка 521 — вызов: `first_audio_chunk=None` (явно)
- Строки 150-153 — проверка и отправка (но никогда не выполняется, так как `first_audio_chunk` всегда `None`)

### `next_message`

**Статус:** Читается в строке 490 для условия логирования (`if error_reason and next_message is None:`). Значение `{"type": "stop"}` (строка 170) никогда не используется для управления потоком — только для логирования. Это мёртвое значение, но не мёртвая переменная.

**Доказательство:** `sed -n '485,495p' app/gateway.py`:

```python
                                    logger.info(
                                        f"[{request_id}] Utterance completed: TTFP={ttfp_ms:.0f}ms" if ttfp_ms else f"[{request_id}] Utterance completed",
                                        extra={"request_id": request_id, "event": "rep_completed"}
                                    )
                                    if error_reason and next_message is None:
                                        logger.error(
                                            f"[{request_id}] Utterance failed: {error_reason}",
                                            extra={"request_id": request_id, "event": "utterance_error", "error": error_reason},
                                        )
                                except asyncio.TimeoutError:
```

### `upstream_close_reason`

**Статус:** Используется только для логирования.

**Доказательство:** `grep -n "upstream_close_reason"` — строки 303, 330, 335 (логирование в `finally`).

### `partial_count`, `ttfp_ms`, `ttft_ms`

**Статус:** Используются для метрик и логирования.

**Доказательство:**
- `partial_count` — строки 92, 218, 316, 340 (инкремент, логирование, возврат)
- `ttfp_ms` — строки 90, 206-213, 226, 314, 340 (TTFP метрика)
- `ttft_ms` — строки 91, 222-229, 315, 340 (TTFT метрика)

**Не мёртвый код — используются для observability.**

---

## 8. Контракты для SDK

### Сообщения от Server к Client

| Тип | Стабилен? | Формат | Строки |
|-----|-----------|--------|--------|
| `ready` | Да | `{"type": "ready", "model": "gigaam-v3-rnnt", "sample_rate": 16000, "version": "2.22.0"}` | 415-421 |
| `partial` | Да | `{"type": "partial", "text": "..."}` (проксируется от upstream) | 205-219 |
| `final` | Да | `{"type": "final", "text": "...", "timings": {...}}` | 232-237 |
| `error` | Реализован (Gap 1 закрыт, ADR-0014) | 4 кода: pool_unavailable, upstream_connect_failed, upstream_ready_timeout, upstream_recv_timeout |

### Сообщения от Client к Server

| Тип | Стабилен? | Формат | Строки |
|-----|-----------|--------|--------|
| `websocket.receive` (binary) | Да | PCM16 аудио байты | 505-527 |
| `end_of_utterance` | Да | `{"type": "end_of_utterance"}` | 476-501 |
| `stop` | Да | `{"type": "stop"}` | 459-475 |

### Коды закрытия WebSocket

| Код | Значение | Строки |
|-----|----------|--------|
| 4401 | Unauthorized | 382 |
| 4429 | Too many concurrent sessions | 395 |
| 4408 | Idle timeout | 438, 536 |
| 1000 | Normal close (client stop) | 474 (без кода, default) |

**Доказательство:** `grep -n "close.*44\|close.*1000"` — строки 382, 395, 438, 474, 536.

### `{"type": "error", "code": ...}`

**Статус:** Реализован (Gap 1 закрыт, ADR-0014, 2026-10-30).

**Коды:** `pool_unavailable`, `upstream_connect_failed`, `upstream_ready_timeout`, `upstream_recv_timeout`.

**Доказательство:** `tests/test_error_type.py` — 2 passed (`upstream_connect_failed`, `pool_unavailable`), 2 manual skipped. Сессия НЕ закрывается при in-session error (кроме idle/limit/auth). См. §12 «Gap 1» ниже.

### Контракты для SDK — Итоговая Таблица

| Контракт | Стабилен? | Что нужно для SDK |
|----------|-----------|-------------------|
| `ready` | Да | Document format |
| `partial` | Да | Document format |
| `final` (с timings) | Да | Document timings fields |
| `end_of_utterance` | Да | Document as sentinel |
| `stop` | Да | Document as session close |
| Коды закрытия (4401, 4408, 4429, 1000) | Да | Document all codes |
| `{"type": "error", "code": ...}` | Да (ADR-0014) | Document 4 error codes; parse on client side |

---

## 9. Обработка Ошибок

### Таблица сценариев ошибок

| Сценарий | Что происходит | Что видит клиент | Строки |
|----------|---------------|------------------|--------|
| `websockets.connect` к gigastt упал | `error_reason = str(e)`, `rep_task` возвращает `error_reason`, upstream закрывается в `except` | `{"type": "error", "code": "upstream_connect_failed"}` (ADR-0014) | 273-280 |
| `upstream.recv()` timeout (30s) | `error_reason = "upstream_recv_timeout"`, цикл прерывается, `upstream_to_client` завершается | `{"type": "error", "code": "upstream_recv_timeout"}` (ADR-0014). Если был partial — клиент видит последний partial, затем error | 189-192 |
| `upstream.close()` timeout (5s) | `logger.warning`, принудительное закрытие через `transport.close()` | Ничего — это внутренняя операция закрытия | 321-327 |
| `websocket.receive()` timeout (idle) | `session_close_reason = "idle_timeout"`, `websocket.close(code=4408)` | WebSocket закрывается с кодом 4408 | 438, 536 |
| `WebSocketDisconnect` в середине реплики | `session_close_reason = "client_disconnect"`, очистка `active_sessions`, логирование | Клиент уже отключился — ничего не видит | 539-545 |
| `upstream_ready` timeout (10s, retry) | `error_reason = "upstream_ready_timeout"`, upstream закрывается, `rep_task` возвращает error | `{"type": "error", "code": "upstream_ready_timeout"}` (ADR-0014) | 131-142 |

### Реализован ли `{"type": "error", "code": ...}` клиенту?

**Да, реализован** (Gap 1 закрыт, ADR-0014, 2026-10-30).

**Доказательство:** 4 места отправки error-сообщений в `app/gateway.py`; `tests/test_error_type.py` — 2 passed. Клиент получает `{"type": "error", "code": ..., "message": ...}`, сессия остаётся открытой (кроме idle/limit/auth), может повторить реплику.

**Для SDK:** `pool_unavailable` и `upstream_ready_timeout` — retry-able (backoff); `upstream_connect_failed` — retry-able с большим backoff; `upstream_recv_timeout` — retry-able, при повторах — сообщить. Парсинг: Python — `_translate_connection_closed` (close codes) + handler in-session `{"type": "error"}`; TS — `translateCloseCode` (close codes) + `onError` (in-session).

---

## 10. Конкурентность и Barge-In

### Вопрос 1: Если `current_rep_task` ещё выполняется, а приходит новый `bytes` — что происходит?

**Ответ:** Аудио идёт в **старую очередь**.

**Доказательство:** `sed -n '500,540p' app/gateway.py`:

```python
                elif "bytes" in message:
                    if audio_queue is None:
                        # First chunk of new utterance - create queue and task
                        rep_counter += 1
                        rep_id = f"rep-{rep_counter:03d}"
                        rep_start_time = time.perf_counter()
                        audio_queue = asyncio.Queue()

                        logger.info(
                            f"[{request_id}] [{rep_id}] Starting utterance processing",
                            extra={"request_id": request_id, "rep_id": rep_id, "event": "utterance_started"},
                        )

                        # Process utterance as background task - allows main loop to handle end_of_utterance
                        rep_task = asyncio.create_task(
                            process_utterance(
                                websocket, audio_queue, session_id, request_id, rep_id, rep_start_time, first_audio_chunk=None
                            )
                        )
                        current_rep_task = rep_task

                    # All chunks (including first) go to the same queue
                    await audio_queue.put(message["bytes"])
```

**Логика:**
- Если `audio_queue is None` — создаётся новая очередь и новый `process_utterance`
- Если `audio_queue is not None` (старая реплика ещё идёт) — аудио кладётся в **старую** очередь

**Вывод:** Новые аудио-чанки не прерывают текущую реплику — они накапливаются в старой очереди, но client_to_upstream уже завершился, новые чанки не читаются, аудио теряется (см. Gap 2).

### Вопрос 2: Если `end_of_utterance` приходит, пока `current_rep_task` не завершился — что происходит?

**Ответ:** Ждём завершения `current_rep_task` (timeout 20 сек), затем сбрасываем очередь и задачу.

**Доказательство:** Строки 476-501:

```python
elif data.get("type") == "end_of_utterance":
    # Put sentinel in queue to signal end_of_utterance
    if audio_queue is not None:
        await audio_queue.put(None)
    # Wait for rep_task to complete and get results
    if current_rep_task and not current_rep_task.done():
        try:
            error_reason, ttfp_ms, ttft_ms, partial_count, next_message, final_text = await asyncio.wait_for(
                current_rep_task, timeout=20.0
            )
            ...
        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] rep_task timeout after end_of_utterance")
            current_rep_task.cancel()
    # Reset queue and task
    audio_queue = None
    current_rep_task = None
    continue
```

**Логика:**
1. Кладем sentinel `None` в очередь (чтобы `client_to_upstream` увидел end_of_utterance)
2. Ждём `current_rep_task` (timeout 20 сек)
3. Сбрасываем `audio_queue = None`, `current_rep_task = None`

### Вопрос 3: Можно ли прервать ассистента (barge-in) или только последовательный диалог?

**Ответ:** **Barge-in НЕ поддерживается.**

**Обоснование:**
1. Аудио-чанки во время выполнения `current_rep_task` идут в старую очередь (не прерывают текущую реплику)
2. `end_of_utterance` ждёт завершения `current_rep_task` (timeout 20 сек) перед началом новой
3. Нет механизма отмены текущей реплики и начала новой "на лету"
4. Нет обработки микрофона на стороне клиента во время воспроизведения ответа ассистента

**Пробел для SDK:** Для barge-in потребуется:
- Механизм принудительной отмены `current_rep_task` при начале новой реплики
- Очистка старой очереди и создание новой без ожидания завершения предыдущей
- Клиентская поддержка детекции начала речи (VAD) для отправки `stop` + нового аудио

---

## 11. Сырые Выводы Тестов

### Auth (tests/test_auth.py)

```bash
$ pytest tests/test_auth.py -v 2>&1 | grep -E "PASSED|FAILED" | head -10
tests/test_auth.py::TestApiKeyAuth::test_valid_api_key PASSED            [ 14%]
tests/test_auth.py::TestApiKeyAuth::test_invalid_api_key PASSED          [ 28%]
tests/test_auth.py::TestApiKeyAuth::test_missing_token PASSED            [ 42%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_ws_query_param PASSED [ 57%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_ws_authorization_header PASSED [ 71%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_http_authorization PASSED [ 85%]
tests/test_auth.py::TestTokenExtraction::test_extract_from_http_no_auth PASSED [100%]
```

### REST (tests/test_rest.py)

```bash
$ pytest tests/test_rest.py -v 2>&1 | grep -E "PASSED|FAILED" | head -6
tests/test_rest.py::TestRestEndpoint::test_transcribe_invalid_wav PASSED [ 25%]
tests/test_rest.py::TestRestEndpoint::test_transcribe_file_too_large PASSED [ 50%]
tests/test_rest.py::TestRestEndpoint::test_transcribe_success PASSED     [ 75%]
tests/test_rest.py::TestRestEndpoint::test_transcribe_upstream_error PASSED [100%]
```

### Backpressure (scripts/test_backpressure.py)

```bash
$ python scripts/test_backpressure.py 2>&1 | grep -E "4429|pool_available|PASS" | head -5
  21-я сессия: closed:4429
  PASS: Получен код 4429 (Too Many Sessions)
  21-я сессия отклонена с кодом 4429: PASS
  ws_active_sessions = 20: PASS
  ws_sessions_rejected_total{reason="limit"} >= 1: PASS
```

### Метрики

```bash
$ grep -n "stt_gateway_" app/metrics.py
18:    "stt_gateway_ws_active_sessions",
24:    "stt_gateway_ws_sessions_total",
31:    "stt_gateway_ws_sessions_rejected_total",
38:    "stt_gateway_ws_session_duration_seconds",
47:    "stt_gateway_partials_received_total",
54:    "stt_gateway_ws_time_to_first_partial_seconds",
61:    "stt_gateway_ws_time_to_final_seconds",
69:    "stt_gateway_rest_requests_total",
76:    "stt_gateway_rest_request_duration_seconds",
84:    "stt_gateway_upstream_errors_total",
```

---

## 12. Known gaps for SDK

### Gap 1: нет структурированных ошибок

**Статус:** Закрыт (ADR-0014, 2026-10-30).

Клиент **получает** `{"type": "error", "code": ...}` при:
- `websockets.connect` к gigastt упал → `upstream_connect_failed`
- `upstream_ready` timeout (10s) → `upstream_ready_timeout`
- `upstream.recv` timeout (30s) → `upstream_recv_timeout`
- pool exhausted → `pool_unavailable`

**Реализация:**
- `app/gateway.py` — 4 места отправки error сообщений (строки 121-128, 146-154, 179-189, 240-248)
- `app/config.py` — `GIGASTT_READY_URL` для проверки доступности pool
- `tests/test_error_type.py` — 2 автоматических теста (upstream_connect_failed, pool_unavailable)

**Клиент видит:**
- Структурированную ошибку с кодом и сообщением
- Возможность реализовать retry с backoff
- Различение временных ошибок (pool_unavailable) от постоянных

**Документация:**
- `docs/api/websocket.md` — таблица error codes (раздел «Ошибки»)
- `docs/api/errors.md` — список error codes
- `docs/adr/0014-gateway-error-messages.md` — архитектурное решение

### Gap 2: overlapping utterances теряют аудио

Если `bytes` приходят, пока `current_rep_task` предыдущей реплики ещё выполняется:
- `audio_queue is None` = False (очередь не сброшена до завершения `current_rep_task`)
- `bytes` идут в **старую** очередь
- `client_to_upstream` старой реплики **уже получил `None`** и сделал `break`
- Новые `bytes` **не читаются**, очередь собирается GC — **аудио потеряно**
- Клиент ждёт `final` для новой фразы — его нет, висит

**Для SDK:** пользователь не может перебить ассистента (barge-in), и не может говорить с паузами <1 сек между фразами — вторая фраза потеряется.

**Требование к to-be:** либо явно **запретить** в `docs/api/websocket.md` («клиент должен дождаться `final` перед отправкой новой реплики»), либо **реализовать** barge-in (новая `audio_queue` при новом `bytes`, отмена `current_rep_task`).

---

## Приложения

### A. grep-команды для верификации

```bash
# Модули и функции
ls app/*.py
grep -n "^async def\|^def\|^class" app/*.py

# State machine
sed -n '400,540p' app/gateway.py  # Главный цикл
sed -n '68,340p' app/gateway.py   # process_utterance

# Read/Write операции
grep -n "websocket.receive\|upstream.recv\|websocket.send_text\|audio_queue.put\|audio_queue.get\|process_utterance" app/gateway.py

# Таймауты
grep -n "timeout\|wait_for\|SESSION_IDLE" app/gateway.py app/config.py

# Мёртвый код
grep -n "final_received\|first_audio_chunk\|next_message\|upstream_close_reason" app/gateway.py

# Контракты
grep -n "type.*ready\|type.*partial\|type.*final\|type.*error\|close.*44" app/gateway.py

# Метрики
grep -n "stt_gateway_" app/metrics.py
```

---

**Конец документа.**
