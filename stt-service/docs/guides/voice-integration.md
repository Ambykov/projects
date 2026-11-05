

# Голосовая интеграция: STT + LLM (и будущий TTS)

**Что описывает:** типичный пайплайн `микрофон/файл → STT → текст → LLM → ответ` и как его собрать из наших сервисов.

**В будущем:** `TTS` для озвучки ответа LLM (Stage 3, отложено).

**Связанные гайды:** [`sdk-python.md`](sdk-python.md), [`sdk-typescript.md`](sdk-typescript.md), `docs/ops/voice-core-scaling.md`, `11.01.00` (LLM-стек), `11.04.00` (voice-core).


## 1. Общая схема

```
┌──────────────┐
│  Микрофон    │  (VAD на клиенте — planned 4.C)
└──────┬───────┘
       │ PCM16 @ 16 kHz mono
       ▼
┌──────────────────────┐
│  STT (voice-core)    │  ws://stt-gateway:8000/ws/stt
│  → partial/final     │
└──────┬───────────────┘
       │ text
       ▼
┌──────────────────────┐
│  LLM (LiteLLM)       │  http://litellm-proxy:4000/v1/chat/completions
│  Qwen3.5 / Qwen3.8   │
└──────┬───────────────┘
       │ text (answer)
       ▼
┌──────────────────────┐
│  TTS (Stage 3)       │  planned — tts-gateway
│  → audio             │
└──────────────────────┘
```

**Текущий статус:** STT и LLM работают. TTS — planned (ADR-0009).


## 2. STT (voice-core)

**Endpoint:** `ws://stt-gateway:8000/ws/stt` (внутри `infra_public_net`) или `ws://localhost:8100/ws/stt` (через SSH-туннель).

**API_KEY:** получить у администратора voice-core. Формат 64 hex, хранение в KeePass, ротация ежеквартально. **Не в git, не в JS-бандле.**

**Быстрый пример (Python):**

```
from stt_client import STTClient, WavAudioSource

client = STTClient(
    base_url="ws://stt-gateway:8000/ws/stt",
    token=os.environ["STT_API_KEY"],
)
result = await client.stream(WavAudioSource("audio.wav"))
text = result.text
```

**Быстрый пример (TypeScript):**

```
import { Session } from '@company/stt-client';

const session = new Session({
  baseUrl: 'ws://stt-gateway:8000/ws/stt',
  token: apiKeyFromBackend,
  workletPath: '/resample-worklet.js',
});
await session.connect();
await session.startUtterance();
// ... пользователь говорит ...
const result = await session.endUtterance();
```

**Ограничения STT:**

- `pool=4` — 4 параллельные реплики, ~16 utt/s (ADR-0021).

- При перегрузке — `pool_unavailable`. См. `docs/ops/voice-core-scaling.md` (pool 4→6→8).

- TTFT линейно растёт с длиной фразы (ADR-0016). Для длинных фраз — VAD (planned) или сегментация на клиенте.


## 3. LLM (LiteLLM)

**Endpoint:** `http://litellm-proxy:4000/v1` (внутри `infra_public_net`) или `http://192.168.0.45:4000/v1` (с хоста).

**API_KEY:** `sk-master-...` (master) или персональный ключ (запросить у администратора LLM). Мастер-ключ — только для администрирования, не для прода.

**Модели:**

- `qwen3.5-122b-fp8` — тяжёлая, для сложных задач.

- `qwen3.8-27b-fp8` — лёгкая, для быстрых ответов.

**Формат:** OpenAI-совместимый (`/v1/chat/completions`).

**Быстрый пример (Python):**

```
from openai import OpenAI

llm = OpenAI(
    base_url="http://litellm-proxy:4000/v1",
    api_key=os.environ["LLM_API_KEY"],
)
response = llm.chat.completions.create(
    model="qwen3.8-27b-fp8",
    messages=[
        {"role": "system", "content": "Ты — голосовой ассистент. Отвечай кратко."},
        {"role": "user", "content": stt_text},
    ],
    max_tokens=500,
)
answer = response.choices[0].message.content
```

**Важно:**

- `no_proxy` должен содержать `litellm-proxy` и IP хоста (см. `09.03`), иначе `403`.

- `max_tokens` — указывать явно, иначе модель может ограничить ответ.

- `chat_template_kwargs: {"enable_thinking": false}` — если нужен «чистый» ответ без reasoning (см. `11.01.03`).


## 4. Полный пример: STT → LLM (Python)

```
import asyncio
import os
from openai import OpenAI
from stt_client import STTClient, WavAudioSource, STTError

async def voice_to_answer(audio_path: str) -> str:
    # 1. STT
    stt = STTClient(
        base_url="ws://stt-gateway:8000/ws/stt",
        token=os.environ["STT_API_KEY"],
    )
    try:
        stt_result = await stt.stream(WavAudioSource(audio_path))
    except STTError as e:
        if e.code == "pool_unavailable":
            return "Сервис распознавания перегружен, повторите"
        raise

    # 2. LLM
    llm = OpenAI(
        base_url="http://litellm-proxy:4000/v1",
        api_key=os.environ["LLM_API_KEY"],
    )
    response = llm.chat.completions.create(
        model="qwen3.8-27b-fp8",
        messages=[
            {"role": "system", "content": "Ты — голосовой ассистент. Отвечай кратко."},
            {"role": "user", "content": stt_result.text},
        ],
        max_tokens=300,
    )

    # 3. (Future) TTS — пока просто возвращаем текст
    return response.choices[0].message.content

print(asyncio.run(voice_to_answer("audio.wav")))
```

**Пример (TypeScript, браузер):**

```
import { Session, STTError } from '@company/stt-client';

async function voiceToAnswer(audioChunks: ArrayBuffer[]): Promise<string> {
  // 1. STT (buffered API — с retry)
  const session = new Session({
    baseUrl: 'ws://stt-gateway:8000/ws/stt',
    token: apiKey,
    workletPath: '/resample-worklet.js',
  });
  const stt = await session.sendUtterance(audioChunks);

  // 2. LLM (через ваш бэкенд — не напрямую, чтобы скрыть LLM_API_KEY)
  const response = await fetch('/api/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: stt.text }),
  });
  const { answer } = await response.json();

  return answer;
}
```

**Рекомендация:** браузерный фронт **не должен** ходить к LiteLLM напрямую — `LLM_API_KEY` утечёт. Ваш бэкенд проксирует: фронт → ваш бэкенд → LiteLLM.


## 5. TTS (planned, Stage 3)

**Статус:** не реализовано. ADR-0009 (архитектура), `11.04.00` §7 (планы).

**План:**

- `tts-model` — sidecar по той же схеме, что `gigastt` (`network_mode: service:stt-gateway`).

- `tts-gateway` — расширение `stt-gateway` (переименование `voice-gateway`).

- `POST /v1/synthesize` (REST) и `/ws/tts` (streaming).

- Модели-кандидаты: MOSS-TTS-Nano (Apache 2.0), kokoro-ru (OpenRAIL). Выбор — после Stage 3.

**Когда:** после VAD (4.C) и передачи SDK (сейчас).


## 6. API-ключи — процесс

### STT (voice-core)

- **Кто выдаёт:** администратор voice-core.

- **Как:** `openssl rand -hex 32` → сохранить в KeePass → передать потребителю защищённым каналом.

- **Хранение потребителем:** env-переменная (через `.env` в compose) или secrets-менеджер. **Не в git, не в JS-бандле, не в логах.**

- **Ротация:** ежеквартально (см. `11.04.01`). При ротации — уведомить всех потребителей за 1 неделю.

- **Отзыв:** при компрометации — сменить в `.env` voice-core, передеплой. Старый ключ → 401.

### LLM (LiteLLM)

- **Кто выдаёт:** администратор LLM-стека (см. `11.01.02`).

- **Master-ключ:** только для администрирования, **не для прода**.

- **Персональные ключи:** с лимитами (RPM/TPM), срок 90 дней.

- **Ротация:** ежемесячно.

### Чек-лист для потребителя

- [ ] Получен `STT_API_KEY` (64 hex) через защищённый канал

- [ ] Ключ в KeePass / Vault, не в git

- [ ] В Docker Compose — через `.env` + `environment:`, не `docker run -e`

- [ ] `no_proxy` содержит `stt-gateway`, `litellm-proxy`, IP хоста

- [ ] Обработка `pool_unavailable` (retry + UI-сообщение)

- [ ] Обработка `unauthorized` (алерт админу)

- [ ] Знает про `docs/ops/voice-core-scaling.md` (если регулярный `pool_unavailable`)


## 7. Частые проблемы при интеграции

| Симптом | Причина | Решение |
| - | - | - |
| `403 Forbidden` из контейнера | `no_proxy` без `stt-gateway` | Добавить в `no_proxy` (см. `09.03`) |
| `pool_unavailable` регулярно | pool=4 исчерпан | Scaling (pool 4→6→8), `docs/ops/voice-core-scaling.md` |
| LLM отвечает обрезанно | `max_tokens` мал | Увеличить `max_tokens` |
| LLM ходит «не туда» (через прокси) | `no_proxy` для `litellm-proxy` | Добавить в `no_proxy` |
| `getUserMedia` не работает | HTTP (не localhost) | HTTPS или `localhost` |
| STT пустой `text` | Тишина / фраза <1s | ADR-0015 — GigaSTT плохо на коротких |



## 8. Ссылки

- [`sdk-python.md`](sdk-python.md) — Python SDK.

- [`sdk-typescript.md`](sdk-typescript.md) — TypeScript SDK.

- `docs/ops/voice-core-scaling.md` — масштабирование STT.

- `11.01.00` — LLM-стек (паспорт).

- `11.01.02` — LLM роли и API-ключи.

- `11.01.03` — LLM инструкции (примеры API).

- `11.04.00` — voice-core (паспорт).

- `11.04.01` — voice-core (администрирование, ротация ключей).

- `09.03` — сетевая безопасность (`no_proxy`).

- ADR-0009 — TTS архитектура (planned).

- ADR-0016 — TTFT на длинных фразах.


