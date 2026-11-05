# @company/stt-client (TypeScript) — v0.2.2

TypeScript SDK для STT Gateway в браузере.

**Что нового в v0.2.2:**

- **`getUserMedia` ошибки типизированы** — `permission_denied`, `microphone_not_found`, `get_user_media_failed` вместо сырого `DOMException`.
- **`AudioWorklet` load ошибка типизирована** — `worklet_load_failed` с путём.
- **`connect(timeoutMs=10000)`** — если `ready` не пришёл за 10 сек, `STTError('connection_timeout')`. Настраивается.
- **`SDK_VERSION` const** — `export const SDK_VERSION = '0.2.2'` (паритет с Python `__version__`).

**Что нового в v0.2.0:**

- **BREAKING:** `SessionOptions.workletPath` — обязателен (убран hardcoded `/static/resample-worklet.js`).
- **`RetryPolicy`** — exponential backoff (1s/2s/4s) для buffered API.
- **`sendUtterance(chunks)`** — buffered API с auto-retry (для файлов, batch, тестов).
- **Vitest** + 17 unit tests (`STTError`, `RetryPolicy`, `translateCloseCode`).
- `prepublishOnly: "npm run build"` — защита от публикации устаревшего `dist/`.

## Установка

Три варианта — в зависимости от того, как передан SDK:

```bash
# 1. Из npm registry (если опубликован)
npm install @company/stt-client

# 2. Из tarball (передан разработчикам как файл)
npm install /path/to/company-stt-client-0.2.2.tgz

# 3. Из локального git-репозитория
npm install ./sdk/typescript
```

## Выбор SDK

Какой SDK использовать в зависимости от сценария:

| Сценарий | SDK | Почему |
|---|---|---|
| **Browser: микрофон** | **TypeScript SDK** | `AudioWorklet` + `getUserMedia`, PCM16 48→16 кГц |
| **Browser: prepared audio** | **TypeScript SDK** (`sendUtterance`) | Buffered API + retry, PCM16 `ArrayBuffer[]` |
| **Node.js / backend** | **Raw WebSocket** | SDK browser-only; шаблон — `examples/node/transcribe.mjs` |
| **Backend batch / CLI** | **Python SDK** | `WavAudioSource`, retry, простой API |
| **>100 req/s** | Raw WS + пул | SDK over-engineered, см. `docs/ops/voice-core-scaling.md` |

**Не смешивайте:** TS SDK не работает в Node.js (нет `AudioWorklet`, `navigator.mediaDevices`), Python SDK не работает в браузере (нет WS DOM API + `sounddevice`).

## workletPath — обязателен с v0.2.0

AudioWorklet для ресемплинга 48→16 кГц. Файл `resample-worklet.js` лежит в пакете: `dist/resample-worklet.js`.

### Три способа указать путь

**1. Копирование в `public/` — универсальный, работает с любым сборщиком (рекомендуем):**

```bash
cp node_modules/@company/stt-client/dist/resample-worklet.js public/
```

```ts
const session = new Session({ baseUrl, token, workletPath: "/resample-worklet.js" });
```

**2. Vite / esbuild / Rollup** — через `?url` suffix:

```ts
import workletUrl from "@company/stt-client/dist/resample-worklet.js?url";
const session = new Session({ baseUrl, token, workletPath: workletUrl });
```

**3. Из git-репозитория SDK (только dev/демо):**

```ts
// файл лежит рядом с examples/web_demo.html
const session = new Session({ baseUrl, token, workletPath: "./examples/resample-worklet.js" });
```

Без `workletPath` — `Session.startUtterance()` бросит `worklet_path_missing`.

## Быстрый старт (streaming с микрофона)

```ts
import { Session, STTError } from '@company/stt-client';

const session = new Session({
  baseUrl: 'ws://localhost:8100/ws/stt',
  token: 'your-api-key',
  workletPath: './resample-worklet.js',  // REQUIRED (v0.2.0+)
  onPartial: (text) => console.log('partial:', text),
  onFinal: (r) => console.log('final:', r.text, r.timings),
  onError: (e) => console.error('error:', e.code, e.isRetryable),
});

await session.connect();

// Реплика 1
await session.startUtterance();       // workletPath из opts
// ... пользователь говорит ...
const r1 = await session.endUtterance();

// Реплика 2 (long-lived соединение)
await session.startUtterance();
const r2 = await session.endUtterance();

await session.close();
```

## Buffered API (prepared audio, с retry)

```ts
import { Session, RetryPolicy } from '@company/stt-client';

const session = new Session({
  baseUrl: 'ws://localhost:8100/ws/stt',
  token: 'your-api-key',
  workletPath: './resample-worklet.js',
  retryPolicy: new RetryPolicy({ maxAttempts: 3 }),  // optional, default 3
});

// chunks: ArrayBuffer[] — PCM16 @ 16 kHz mono
const result = await session.sendUtterance(chunks, (p) => console.log('partial:', p));
console.log(result.text, result.timings);
```

**Разница между API:**

| Метод | Retry | Когда |
| - | - | - |
| `startUtterance()` / `endUtterance()` | ❌ нет | Real-time микрофон. `pool_unavailable` → аудио потеряно → caller показывает «повторите» пользователю |
| `sendUtterance(chunks)` | ✅ 3× (1s/2s/4s) | Prepared audio (файлы, batch, тесты). Тот же набор chunks на каждой попытке |


## API

### Session

```ts
class Session {
  constructor(opts: SessionOptions)
  connect(timeoutMs?: number): Promise<void>       // default 10000 ms
  startUtterance(workletPath?: string): Promise<void>  // override для одной реплики
  endUtterance(): Promise<FinalResult>
  sendUtterance(chunks: ArrayBuffer[], onPartial?: (text: string) => void): Promise<FinalResult>
  close(): Promise<void>
}
```

### SessionOptions

```ts
interface SessionOptions {
  baseUrl: string;
  token: string;
  /** REQUIRED: URL/path к AudioWorklet модулю (resample-processor). */
  workletPath: string;
  /** Optional override для buffered API. Default: 3 attempts, 1s/2s/4s. */
  retryPolicy?: RetryPolicy;
  onPartial?: (text: string) => void;
  onFinal?: (result: FinalResult) => void;
  onError?: (err: STTError) => void;
  onReady?: () => void;
}
```

### RetryPolicy

```ts
class RetryPolicy {
  constructor(opts?: { maxAttempts?: number; baseDelayMs?: number; maxDelayMs?: number });
  delay(attempt: number): number;                     // 1s, 2s, 4s, 8s... (capped)
  run<T>(fn: () => Promise<T>): Promise<T>;           // retry на retryable STTError
}
```

### STTError

```ts
class STTError extends Error {
  code: string;
  message: string;
  isRetryable: boolean;
  constructor(code: string, message: string, retryableOverride?: boolean);
}
```

## Ошибки

`STTError.code` — один из:

**Retryable (автоматически для `sendUtterance`):**

- `pool_unavailable` — нет свободных слотов модели (retry через backoff)
- `upstream_ready_timeout` — gateway не получил ready от модели за 10s
- `upstream_recv_timeout` — gateway не получил ответ за 30s
- `upstream_connect_failed` — не удалось подключиться к upstream
- `session_limit` — close code 4429 (лимит `MAX_CONCURRENT_SESSIONS`)
- `idle_timeout` — close code 4408
- `connection_timeout` — `connect()` не дождался `ready` за `timeoutMs` (v0.2.2)

**Not retryable:**

- `unauthorized` — close code 4401 (неверный API-ключ)
- `protocol_error`, `client_timeout`, `ws_not_open`, `ws_error`
- `connection_closed` — unknown close codes (код в message)
- **`permission_denied`** — пользователь запретил доступ к микрофону (v0.2.2)
- **`microphone_not_found`** — в системе нет микрофона (v0.2.2)
- **`get_user_media_failed`** — прочие ошибки `getUserMedia` (v0.2.2)
- **`worklet_load_failed`** — не удалось загрузить `resample-worklet.js` по `workletPath` (v0.2.2)
- **`worklet_path_missing`** — `workletPath` не передан в `SessionOptions`
- **`ws_close_N`** — **удалено в v0.2.1**, заменено на `connection_closed`

## Обработка перегрузки (streaming API)

`pool_unavailable` в streaming API означает, что аудио уже потеряно. Рекомендуемый паттерн:

```ts
try {
  await session.startUtterance();
  // ... пользователь говорит ...
  const result = await session.endUtterance();
} catch (e) {
  if (e instanceof STTError && e.code === 'pool_unavailable') {
    showUserMessage('Сервис перегружен, повторите фразу');
    // ⚠️ Это сигнал команде эксплуатации — см. docs/ops/voice-core-scaling.md
  }
}
```

## Web Demo

```bash
cd sdk/typescript
npm install
npm run build

# Запустить статический сервер в корне sdk/typescript/
# (web_demo.html импортирует ../dist/index.js, workletPath — ./resample-worklet.js рядом)
python3 -m http.server 8000

# Открыть http://localhost:8000/examples/web_demo.html
```

**Примечание:** `examples/web_demo.html` использует `workletPath: './resample-worklet.js'` — файл лежит рядом в `examples/`. Для прода — свой путь к worklet (`/resample-worklet.js` из `public/` или Vite `?url`).

## Тестирование

```bash
npm test          # 17 tests, 2 файла (errors, retry)
npm run build     # tsc → dist/
```

## Best practices

Проверенные практикой грабли — из реальной интеграции (demo + review).

### ✅ DO

- **`workletPath` — из `public/`** (универсально) или `import workletUrl from '...?url'` (Vite/esbuild/Rollup). Не хардкодить `/static/resample-worklet.js` — 404.
- **`API_KEY` — через свой backend**: фронт → ваш backend → `stt-gateway`. Ключ **никогда** не попадает в бандл или `localStorage`.
- **`onPartial` — замена** (`resultEl.textContent = text`), не накопление (`+=`).
- **`getUserMedia` — только HTTPS или localhost**. `http://192.168.x.x` не сработает.
- **Ждать `final`** перед следующей репликой (overlapping не поддерживается — Gap 2).
- **Обрабатывать `permission_denied`** — показать UI «разрешите доступ к микрофону».
- **`connect(timeoutMs)`** — задать таймаут (default 10s), если сервер может отвечать медленно.
- **Логировать `result.timings`** — единственный способ увидеть реальный TTFT.

### ❌ DON'T

- **Не использовать SDK в Node.js** — только браузер. Для Node.js — `examples/node/transcribe.mjs` (raw WebSocket).
- **Не полагаться на retry в streaming** (`startUtterance`/`endUtterance`) — by design: аудио с микрофона потеряно, повторять нечего. Retry только в `sendUtterance(chunks)` (prepared audio).
- **Не `resultText += partial.text`** — получите «наст наст настоящий наст настоящий свод». Partial это **замена**.
- **Не хардкодить путь worklet** (`/static/resample-worklet.js` — старый путь из v0.1.0, файла там нет).
- **Не отправлять WAV-заголовок** — только raw PCM16 (`ArrayBuffer`).
- **Не отправлять 48 kHz** — SDK ожидает 16 kHz mono (AudioWorklet ресемплирует, если вход 48).
- **Не ждать `partial` на <1 сек** — GigaSTT не даёт (ADR-0005).
- **Не использовать `retryPolicy` для `connect()`** — у `connect()` свой таймаут (`timeoutMs`).

## Ограничения

- **Overlapping utterances не поддерживаются** (Gap 2): `endUtterance()` блокирует `startUtterance()` до final.
- **`getUserMedia` требует localhost или HTTPS** (`wss://`).
- **Аудио**: AudioWorkletNode + ресемплинг 48→16 кГц, PCM16.
- **Browser-only**, не Node (AudioWorklet / WebSocket DOM).
- **Для фраз >5 сек** — TTFT линейный (ADR-0016). Сегментируйте на 3–5-сек куски или ждите VAD (planned).

## Ссылки

- ADR-0014 — Gap 1 (structured errors)
- ADR-0017 — Python SDK MVP
- ADR-0019 — TS SDK v0.1.0 fixes + v0.2.0 + v0.2.1 + v0.2.2
- `docs/api/websocket.md` — контракт WS (ready/partial/final/error/close codes) + sequence diagram + TTFT таблица
- `docs/guides/sdk-typescript.md` — интеграция в наш проект (сервер, workletPath, FAQ)
- `docs/ops/voice-core-scaling.md` — план масштабирования сервиса
