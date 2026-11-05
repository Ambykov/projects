# Архитектура — поток данных

## Основной поток (STT)

```
Микрофон
    │
    ▼
┌─────────────────┐
│   SDK Client    │  Resampling: 48kHz → 16kHz
│   VAD detection │  Fragment audio into chunks
└────────┬────────┘
         │ WebSocket (binary PCM16)
         ▼
┌─────────────────┐
│  Voice Gateway  │  Auth, rate limiting
│  /ws/stt        │  Per-utterance upstream
└────────┬────────┘
         │ WebSocket upstream
         ▼
┌─────────────────┐
│    GigaSTT      │  Inference ~110ms (сервер, ADR-0020)
│  ws://gigastt   │  Partial streaming
└────────┬────────┘
         │
         ▼
    partial* → final
         │
         ▼
┌─────────────────┐
│  Voice Gateway  │  Forward to client
└────────┬────────┘
         │
         ▼
    SDK Client
```

## Задержки

| Этап | Local baseline | Server baseline (ADR-0020/0021) |
| - | - | - |
| TTFP сервер (upstream\_first\_byte) | 100–300 ms (короткие), 800–2000 ms (длинные) — ADR-0016 | 110–121 ms (1-сек фраза) |
| TTFP клиент (браузер) | +500–700 ms overhead (сеть, ресемплинг) — ADR-0015 | +641 ms (пример из browser test) |
| Инференс GigaSTT (raw) | ~226 ms (RTF 0.226, ADR-0001) | ~110 ms (RTF 0.104–0.113, ADR-0020) |
| TTFT end-to-end (1-сек фраза) | ~226 ms | ~227.5 ms (overhead ~117 ms нивелирует выигрыш, ADR-0020) |
| Upstream connect | 4 ms (ADR-0006) | 14.5 ms (ADR-0020) |


**Ключевой вывод:** raw inference на сервере в 2× быстрее (226 → 110 ms), но end-to-end TTFT не изменился — сетевой и протокольный overhead (~117 ms) доминирует. Throughput вырос за счёт `pool=4` (ADR-0021).

## Многофазовая сессия

```
Client: [аудио 1] → end_of_utterance → [ЖДАТЬ final 1] → [аудио 2] → end_of_utterance → [ЖДАТЬ final 2] → stop
Gateway: ↑upstream1↓                  (connection held)         ↑upstream2↓                  (connection held)
GigaSTT: [final 1]                                              [final 2]
```

Каждая реплика — отдельное upstream-соединение, клиентское соединение остаётся открытым. **Overlapping не поддерживается:** клиент должен дождаться `final` перед отправкой следующей реплики (см. Gap 2 в `gateway-as-is.md`).

