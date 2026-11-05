# ADR-0019: TypeScript SDK v0.1.0 — фиксы и известные ограничения

**Статус:** Accepted **Дата:** 2026-11-02 **Связано:** ADR-0006 (per-utterance), ADR-0014 (Gap 1), ADR-0016 (TTFT), ADR-0017 (Python SDK MVP)

## Контекст

TS SDK `@company/stt-client` v0.1.0 — browser SDK (long-lived Session + AudioWorklet). Ревью выявило 5 блокеров и 5 majors. Часть исправлена в v0.1.0, часть отложена в v0.2.0.

Цель: привести TS SDK в соответствие с контрактом `docs/api/websocket.md` и Python SDK, насколько возможно без переписывания архитектуры.

## Исправлено в v0.1.0 (commit cae5d19)

### B1: `connect()` зависал навсегда при close до ready

**Было:** `onclose` не вызывал `reject` при `!connected`. Неверный токен → 4401 → клиент висит вечно.

**Стало:** в `onclose` — `if (!connected) reject(err)`.

### B2: Close codes не семантические

**Было:** `ws_close_4401`, `ws_close_4429`, `ws_close_4408` — единый `STTError` без семантики.

**Стало:** `translateCloseCode(code, reason)`:

- 4401 → `STTError('unauthorized', ...)`, `isRetryable=false`

- 4429 → `STTError('session_limit', ...)`, `isRetryable=true`

- 4408 → `STTError('idle_timeout', ...)`, `isRetryable=true`

Паритет с Python SDK `_translate_connection_closed()` (ADR-0017).

### M1: `isRetryable` не знал про 4429/4408

**Было:** только 4 error-кода Gap 1.

**Стало:** +`session_limit`, +`idle_timeout`. Паритет с Python `STTError.is_retryable`.

### M2/M3: `ws_not_open` guards

**Было:** `startUtterance`/`endUtterance` не проверяли `ws.readyState` — запись в мёртвый сокет, сырой `Error`.

**Стало:** `if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new STTError('ws_not_open', ...)`.

## Отложено в v0.2.0 (зафиксировано)

### M4: `workletPath` default → 404

**Проблема:** `startUtterance(workletPath = '/static/resample-worklet.js')`. Файл удалён (ADR-0018/коммит 55d0eef). 404.

**Решение v0.2.0:** SDK не должен хардкодить путь. Варианты:

- Требовать явный `workletPath` (обязательный параметр), документация — «укажите свой worklet».

- Или раздавать worklet из npm-пакета (`files: ["dist", "worklet"]`), URL через CDN или `import.meta.url`.

### M5: Нет тестов

`tests/` отсутствует. `package.json` без `test` script. Ноль покрытия.

**Решение v0.2.0:** vitest + unit-тесты:

- `STTError.isRetryable` — whitelist

- `translateCloseCode` — 4 кода + default

- `Session` — connect hang, ws\_not\_open guards, waitingFinal

- Возможно, e2e через `ws` mock

### `resample-worklet.ts` не в `tsconfig.include`

`tsconfig.include` содержит 5 файлов, `resample-worklet.ts` отсутствует. `dist/resample-worklet.js` не обновляется `npm run build`. Сейчас — работает, потому что worklet лежит отдельно в `examples/`. В v0.2.0 — включить в сборку или вынести в `worklet/` с отдельной командой.

### Retry-loop

В Python SDK есть `RetryPolicy` (exponential backoff 1/2/4s). В TS — только `isRetryable` флаг + `onError` callback. Пользователь должен сам реализовать retry.

**Решение v0.2.0:** синхронно с Python `STTSession` (long-lived):

- `Session.sendUtterance()` с retry-loop

- `Session.close()` идемпотентный

- Паритет API

### `prepublishOnly`

Нет `prepublishOnly: "npm run build"` в `package.json`. Риск опубликовать устаревший `dist/`. В v0.2.0 — добавить.

## Что НЕ сделано и почему

- **Тег `sdk-ts-v0.1.0`** — не ставим. v0.1.0 содержит известные ограничения (M4, M5). После v0.2.0 — тег.

- **Retry-loop** — откладывается, чтобы синхронизировать с Python `STTSession` в одном релизе.

- **Тесты** — откладываются, чтобы не растягивать сессию. Покрытие Python SDK (5 тестов) считается достаточным baseline для v0.1.0.

## Проверки

- `tsc --noEmit` → EXIT=0 (strict mode)

- `npm run build` → EXIT=0

- `dist/session.js` → `translateCloseCode` + `ws_not_open` присутствуют

- `dist/errors.js` → `session_limit` + `idle_timeout` присутствуют

- `git status sdk/typescript/` — пусто (`dist/` в `.gitignore`)

## Ссылки

- ADR-0006 — per-utterance upstream reconnection

- ADR-0014 — Gap 1 (structured errors)

- ADR-0017 — Python SDK MVP one connection per utterance

- ADR-0018 — load test results

- `docs/api/websocket.md` — контракт WS

- `sdk/typescript/src/session.ts`, `errors.ts` — код

## v0.2.0: выполнено (2026-11-02)

Все отложенные в v0.1.0 задачи закрыты. Тег `sdk-ts-v0.2.0`.

### Закрыто

| M | Задача | Коммит |
| - | - | - |
| M1 (продолжение) | `RetryPolicy` class + `STTError.retryable` override | `63be41b` |
| M5 | Vitest + 17 unit tests (`STTError`, `RetryPolicy`, `translateCloseCode`) | `a2a0b03` |
| M4 | `workletPath` required в `SessionOptions` (**BREAKING**) | `158ec93` |
| M3 | `resample-worklet.ts` в `tsconfig.include` + `audioworklet.d.ts` (ambient types) | `9cd6a81`, `8611ba7` |
| M6 | `sendUtterance(chunks)` — buffered API с auto-retry | `005f1be` |
| — | `prepublishOnly: "npm run build"` | `9cd6a81` |
| — | `translateCloseCode` вынесен в `errors.ts` (testability) | `cf2ec8b` |


### Архитектурное решение: два API

**Streaming (`startUtterance` / `endUtterance`)** — **без** auto-retry. Причина: аудио захватывается с микрофона и отправляется сразу. При `pool_unavailable` аудио **потеряно** — retry невозможен. Caller (UI) показывает пользователю «повторите фразу», это **сигнал команде эксплуатации** о необходимости масштабирования (см. `docs/ops/voice-core-scaling.md`).

**Buffered (`sendUtterance(chunks)`)** — **с** auto-retry (`RetryPolicy`, 3 попытки, 1s/2s/4s backoff). Для prepared audio: файлы, batch-обработка, тесты.

`connect()` — **без** auto-retry (caller видит `pool_unavailable`, `4429`, `4401`).

### Breaking change

`SessionOptions.workletPath` — обязателен. Раньше был hardcoded `/static/resample-worklet.js` в 3 местах (`audio.ts`, `session.ts`, `web_demo.html`). На сервере файла нет → 404 при каждой реплике.

Миграция v0.1.0 → v0.2.0:

```
// v0.1.0
const session = new Session({ baseUrl, token });
await session.startUtterance('/my/path/worklet.js');

// v0.2.0
const session = new Session({
  baseUrl,
  token,
  workletPath: '/my/path/worklet.js',  // ← переехало в опции
});
await session.startUtterance();  // без аргумента
```

### Проверки

- `npx tsc --noEmit` → `EXIT_TSC=0` (strict)

- `npm test` → **17 passed** (2 файла: errors, retry)

- `npx tsc` → dist/\*.js + \*.d.ts (12 файлов, включая `resample-worklet.js`)

- `git status` — чистый (кроме `?? egg-info`)

### Что не сделано (осознанно)

- **Интеграционный тест в браузере** (Playwright/Puppeteer) — отдельная задача, требует headless browser.

- **`sendUtterance` с `AudioSource` (streaming из файла)** — пока только `ArrayBuffer[]`. Для больших файлов — chunked reading caller-side.

- **`STTSession` (long-lived buffered)** — если появится запрос от пользователей.

### Ссылки

- ADR-0017 — Python SDK MVP (там же паритет)

- ADR-0021 — load test на сервере (throttling, `pool_unavailable` при \>4)

- `docs/ops/voice-core-scaling.md` — план масштабирования


## v0.2.1: паритет error codes (2026-11-03)

Минорное обновление после ревью handoff. Цель — устранить расхождение `STTError.code` между Python и TS SDK.

### Изменено

| Было (v0.2.0) | Стало (v0.2.1) | Коммит |
| - | - | - |
| `translateCloseCode(code)` default: `ws_close_${code}` | `connection_closed` (код в `message`) | `7c970b5` |


**Паритет с Python:** `sdk/python/stt_client/client.py::_translate_connection_closed` возвращает `STTError("connection_closed", f"WebSocket closed: code={code}")`. Теперь TS — идентично (без `reason` в message, для точного совпадения).

### Почему breaking приемлем

- SDK не был передан потребителям (только внутреннее ревью + browser demo).

- `STTError.code` — машинный код для retry-логики, не номер WS close. `ws_close_1006` заставлял caller'а парсить число; `connection_closed` семантичнее.

- Reason из close-frame не включается в message — совместимо с Python (который его тоже не включает).

### Проверки

- `npm run build` → EXIT=0

- `npm test` → 17 passed (9 errors + 8 retry)

- `npm pack` → `company-stt-client-0.2.1.tgz` (8865 байт)

- Smoke: `npm install` из tarball в чистой папке → `worklet на месте`, `import OK`, `translateCloseCode(1006).code == "connection_closed"`

### Тег

`sdk-ts-v0.2.1` — на коммите `7c970b5`.

### Ссылки

- Python паритет: `sdk/python/stt_client/client.py` (строки 18-27)

- `docs/api/errors.md` — раздел «WebSocket error messages»

- ADR-0014 — Gap 1 (structured errors)

