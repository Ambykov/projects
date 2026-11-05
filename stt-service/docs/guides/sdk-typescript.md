


# TypeScript SDK — руководство по интеграции

**Пакет:** `@company/stt-client` v0.2.2 **Полная документация пакета:** [`sdk/typescript/README.md`](../../sdk/typescript/README.md) (API reference, breaking changes v0.2.0 + v0.2.1 error codes + v0.2.2 typed errors, примеры). **Этот гайд** — про интеграцию в **наш проект** (браузер, сервер `serverllm`, `workletPath`, перегрузка).


## 1. Куда подключаться

### Локальная разработка

Gateway на `localhost:8100`:

```
import { Session } from '@company/stt-client';

const session = new Session({
  baseUrl: 'ws://localhost:8100/ws/stt',
  token: 'testtoken',
  workletPath: './resample-worklet.js',
});
```

### Сервер (`serverllm`)

Прод-сервер **не отдаёт STT-порт наружу**. Для браузерной демки — **проблема**: `getUserMedia` требует **secure context** (`https://` или `localhost`), а `ws://serverllm:8100` — не secure, и порт 8100 не проброшен наружу.

**Варианты для разработки:**

| Сценарий | Решение |
| - | - |
| Frontend на ноутбуке разработчика | Gateway в Docker локально (`localhost:8100`), dev-сервер фронта на `localhost:3000` или `:5173`. CORS: добавить в `.env` gateway |
| Frontend на сервере (внутри `infra_public_net`) | Frontend-контейнер в `infra_public_net`, `baseUrl: 'ws://stt-gateway:8000/ws/stt'`, но `getUserMedia` не сработает без `https://` — нужен reverse proxy с TLS |
| Тестирование с ноутбука против сервера | SSH-туннель `ssh -p 2222 -L 8100:127.0.0.1:8100 llmuser@192.168.0.45` + dev-сервер фронта на `localhost` (secure context) |


**Ключевое:** `getUserMedia` работает **только** на `https://` или `http://localhost`. `http://192.168.0.45:3000` — не сработает.

### Внутри `infra_public_net` (Docker-контейнер)

```
baseUrl: 'ws://stt-gateway:8000/ws/stt'
```

**Обязательно:** `stt-gateway` и IP хоста в `no_proxy` — иначе запросы уйдут через корпоративный прокси и получат `403` (см. `09.03`).


## 2. API-ключ

`API_KEY` выдаёт администратор voice-core. Формат: 64 hex-символа.

**Хранение:** KeePass / Bitwarden / Vault. **Не в git, не в JS-бандле, не в `localStorage` в проде.**

```
// Плохо: ключ в бандле
const session = new Session({ token: 'hardcoded' /* ... */ });

// Хорошо: ключ получен с бэкенда (после аутентификации пользователя)
const { token } = await fetch('/api/voice/token').then(r => r.json());
const session = new Session({ token /* ... */ });
```

**Рекомендуемая архитектура:** фронт ходит на ваш бэкенд, бэкенд проксирует к `stt-gateway` с `API_KEY` в header. Это скрывает ключ от браузера.

**Для локальной разработки / демо:** токен вводится в input (как в `examples/web_demo.html`). В прод — **не так**.


## 3. Быстрый старт

```
import { Session, STTError } from '@company/stt-client';

const session = new Session({
  baseUrl: 'ws://localhost:8100/ws/stt',
  token: 'your-api-key',
  workletPath: './resample-worklet.js',   // REQUIRED (v0.2.0+)
  onPartial: (text) => console.log('partial:', text),
  onFinal: (r) => console.log('final:', r.text, r.timings),
  onError: (e) => console.error('error:', e.code, e.isRetryable),
});

await session.connect();

await session.startUtterance();
// ... пользователь говорит ...
const result = await session.endUtterance();
console.log(result.text);

await session.close();
```

**`workletPath` (breaking change v0.2.0):** раньше был hardcoded `/static/resample-worklet.js`. Теперь **обязателен** в `SessionOptions`. Файл `resample-worklet.js` — AudioWorklet, должен быть доступен по указанному URL.

**Где взять `resample-worklet.js`:** соберите из `sdk/typescript/src/resample-worklet.ts` (`npm run build` → `dist/resample-worklet.js`) и положите рядом с вашим `index.html` (или отдайте как static).


## 4. Два API: streaming vs buffered

| Метод | Retry | Когда |
| - | - | - |
| `startUtterance()` / `endUtterance()` | ❌ нет | Real-time микрофон. Пользователь говорит, аудио уходит сразу |
| `sendUtterance(chunks)` | ✅ 3× (1s/2s/4s) | Prepared audio: файл, batch, тест. Chunks — `ArrayBuffer[]` PCM16 @ 16 kHz mono |


### Streaming (микрофон)

```
try {
  await session.startUtterance();
  // ... пользователь говорит ...
  const result = await session.endUtterance();
} catch (e) {
  if (e instanceof STTError && e.code === 'pool_unavailable') {
    // Аудио потеряно (пользователь уже сказал), retry невозможен
    showUserMessage('Сервис перегружен, повторите фразу');
  }
}
```

### Buffered (файл/батч)

```
// chunks: ArrayBuffer[] — PCM16 @ 16 kHz mono
const result = await session.sendUtterance(chunks, (partial) => {
  console.log('partial:', partial);
});
console.log(result.text);
```

**Внутри `sendUtterance`:** на каждой попытке открывается новый WS, отправляются все chunks, ждётся final. `pool_unavailable` и `session_limit` ретраятся прозрачно.


## 5. Обработка `pool_unavailable` (streaming)

`pool_unavailable` в streaming = **аудио потеряно**. Retry SDK не поможет (пользователь уже сказал фразу).

**Рекомендуемый паттерн UI:**

```
try {
  await session.startUtterance();
  const result = await session.endUtterance();
} catch (e) {
  if (e instanceof STTError) {
    if (e.code === 'pool_unavailable') {
      // 3 retry не помогли (для buffered) или аудио потеряно (streaming)
      // ⚠️ Сигнал эксплуатации масштабировать — docs/ops/voice-core-scaling.md
      showUserMessage('Сервис перегружен, повторите');
    } else if (e.code === 'unauthorized') {
      // 4401 — неверный API_KEY, retry бессмыслен
      showUserMessage('Ошибка авторизации, обратитесь к администратору');
    } else if (e.code === 'client_timeout') {
      showUserMessage('Слишком долго, попробуйте короче');
    } else {
      showUserMessage(`Ошибка: ${e.code}`);
    }
  }
}
```

**Команде эксплуатации:** если `pool_unavailable` регулярно (>5% запросов в час) — pool 4→6→8 (5 минут, throughput ×2). См. `docs/ops/voice-core-scaling.md`.


## 6. Web Demo

Готовый пример: `sdk/typescript/examples/web_demo.html`.

```
cd sdk/typescript
npm install
npm run build

# Структура для раздачи (любой static server):
# your-www/
#   ├── index.html           (копия examples/web_demo.html)
#   ├── resample-worklet.js  (копия examples/resample-worklet.js)
#   └── dist/                (копия dist/ — из него импортирует index.html)

# Простой static server:
python3 -m http.server 8000
# Открыть http://localhost:8000
```

**Важно:** в `web_demo.html` путь `import { Session } from '../dist/index.js'` — если вы копируете в другую структуру, поправьте. Плюс `workletPath: './resample-worklet.js'` — файл должен лежать рядом.

`getUserMedia` на `http://localhost:8000` — работает (localhost = secure context).


## 7. Тесты SDK

```
cd sdk/typescript
npm test              # 17 unit tests (errors, retry) — vitest
npm run build         # tsc → dist/
```

**Интеграционные тесты в браузере** (Playwright) — не реализованы. Отдельная задача.


## 8. Частые проблемы

| Симптом | Причина | Решение |
| - | - | - |
| `STTError('unauthorized')` (4401) | Неверный API_KEY | Проверить токен |
| `STTError('pool_unavailable')` | 4 слота заняты | Retry; при регулярных — scaling.md |
| `NotAllowedError` от `getUserMedia` | Не HTTPS и не localhost | Использовать `localhost` или TLS |
| `worklet_path_missing` | Не передан `workletPath` | `new Session({ workletPath: './resample-worklet.js', ... })` |
| 404 на `resample-worklet.js` | Файл не доступен по URL | Скопировать из `dist/` или `examples/` |
| `Cannot read property 'ws' of null` | Использование `session` до `connect()` | `await session.connect()` сначала |
| `TypeError: Failed to construct 'AudioWorkletNode'` | Старый браузер / нет поддержки AudioWorklet | Chrome/Firefox/Edge последних версий |
| `Failed to execute 'addModule'` | Битый worklet или неверный MIME type | Проверить, что файл — валидный JS, отдаётся с `Content-Type: application/javascript` |
| `STTError('permission_denied')` | Пользователь запретил доступ к микрофону | Показать UI «разрешите микрофон в настройках браузера»; повторить `getUserMedia` после разрешения |
| `STTError('microphone_not_found')` | В системе нет микрофона | Проверить, что микрофон подключён; `navigator.mediaDevices.enumerateDevices()` |
| `STTError('worklet_load_failed')` | 404 на `workletPath` | Проверить путь: файл должен быть в `public/` или через Vite `?url` (см. `sdk/typescript/README.md` §workletPath) |
| `STTError('connection_timeout')` | `ready` не пришёл за `timeoutMs` (default 10s) | Увеличить `connect(30000)`; проверить gateway `curl /health` |
| `final` пришёл, но текст **обрезан** на фразе 10–15 сек | `upstream_final = 15s` хардкод gateway | Сегментировать (ADR-0016); VAD planned (ADR-0022) |



## 8.5. Do's and Don'ts (кратко)

Полный список — в `sdk/typescript/README.md` §«Best practices». Здесь — самые частые грабли.

### ✅ DO

- **`workletPath` — из `public/`** (универсально) или Vite `?url`. Не хардкодить `/static/resample-worklet.js` — 404.
- **`API_KEY` — через свой backend**: фронт → ваш backend → `stt-gateway`. Ключ не в бандле, не в `localStorage`.
- **`onPartial` — замена** (`resultEl.textContent = text`), не накопление (`+=`).
- **`getUserMedia` — только HTTPS или localhost**.
- **Ждать `final`** перед следующей репликой (overlapping не поддерживается).
- **Обрабатывать `permission_denied`** — UI «разрешите доступ к микрофону».
- **`connect(timeoutMs)`** — задать таймаут, если сервер отвечает медленно.
- **Для фраз >5 сек — сегментировать** (ADR-0016).

### ❌ DON'T

- **Не использовать SDK в Node.js** — только браузер. Node.js — `examples/node/transcribe.mjs` (raw WebSocket).
- **Не полагаться на retry в streaming** — by design: аудио потеряно.
- **Не `resultText += partial.text`** — partial это замена.
- **Не хардкодить путь worklet** — старый путь из v0.1.0.
- **Не отправлять WAV-заголовок** — только raw PCM16.
- **Не отправлять 48 kHz** — только 16 kHz mono.
- **Не ждать `partial` на <1 сек** — GigaSTT не даёт (ADR-0005).
- **Не использовать `retryPolicy` для `connect()`** — свой `timeoutMs`.


## 9. Миграция v0.1.0 → v0.2.0

**Breaking change:** `workletPath` обязателен.

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

**Новое в v0.2.0:**

- `RetryPolicy` (1s/2s/4s backoff) для buffered API.

- `sendUtterance(chunks)` — buffered API.

- 17 unit tests (vitest).

- `prepublishOnly` (защита от публикации старого `dist/`).


## 10. Ссылки

- `sdk/typescript/README.md` — API reference, примеры, breaking changes.

- `docs/api/websocket.md` — протокол WS (ready/partial/final/error/close codes).

- `docs/api/errors.md` — все коды ошибок и retry-able.

- `docs/ops/voice-core-scaling.md` — план масштабирования.

- ADR-0019 — TS SDK v0.1.0 fixes + v0.2.0 + v0.2.1 (error code parity).

- ADR-0016 — TTFT на длинных фразах.


