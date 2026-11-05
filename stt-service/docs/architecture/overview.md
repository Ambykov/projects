# Архитектура — обзор

## Три слоя

```
┌─────────────────────────────────────────────────────────┐
│                    Models Layer                         │
│  ┌─────────────┐    ┌─────────────┐                    │
│  │   GigaSTT   │    │  TTS Model  │                    │
│  │  v2.22.0    │    │   (planned) │                    │
│  └──────┬──────┘    └──────┬──────┘                    │
└─────────┼──────────────────┼────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Voice Gateway                          │
│  ┌─────────────┐    ┌─────────────┐                    │
│  │   /ws/stt   │    │   /ws/tts   │                    │
│  │   /v1/stt   │    │   /v1/tts   │                    │
│  │   (active)  │    │   (planned) │                    │
│  └─────────────┘    └─────────────┘                    │
│           Auth, Metrics, Backpressure                   │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐
│   SDK (Python)  │    │  SDK (TypeScript)   │
│   v0.1.0        │    │   v0.2.0            │
│   VAD (planned) │    │   VAD (planned)     │
└─────────────────┘    └─────────────────────┘
```

**Сноска:** TTS (Stage 3, отложен), VAD (Stage 4.C, planned) — не реализованы. SDK (Python v0.1.0, TypeScript v0.2.0) — выпущены (ADR-0017, ADR-0019). См. `roadmap.md`.

## Компоненты

### Models Layer

| Компонент | Версия | Описание |
|-----------|--------|----------|
| GigaSTT | 2.22.0 | STT модель (gigaam-v3-rnnt) |
| TTS Model | - | TTS модель (Stage 3) |

### Voice Gateway

| Компонент | Описание |
|-----------|----------|
| WebSocket proxy | Long-lived сессии с per-utterance upstream |
| REST API | /v1/transcribe для file upload |
| Auth | API-key (основной), JWT (заготовка) |
| Metrics | Prometheus для VictoriaMetrics |
| Logging | Structured JSON для Loki |

### SDK

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Python SDK | v0.1.0 (ADR-0017) | `STTClient.stream()` — one connection per utterance; `WavAudioSource` / `MicAudioSource`; `RetryPolicy` (3 попытки, 1s/2s/4s) |
| TypeScript SDK | v0.2.0 (ADR-0019) | `Session` (long-lived WS) + AudioWorklet; `startUtterance/endUtterance` (streaming, без retry) и `sendUtterance(chunks)` (buffered, с retry) |
| Resampling | в SDK | 48kHz → 16kHz: Python — в `WavAudioSource`/`MicAudioSource`, TS — AudioWorklet (`resample-worklet.js`) |
| VAD client | planned (4.C) | Определение конца фразы |

См. [`sdk/python/README.md`](../../sdk/python/README.md) и [`sdk/typescript/README.md`](../../sdk/typescript/README.md).

## Ссылки на ADR

- [ADR-0006](../adr/0006-per-utterance-upstream-reconnection.md) — Per-utterance upstream reconnection
- [ADR-0008](../adr/0008-docker-verification.md) — Docker verification
- [ADR-0012](../adr/0012-single-reader-queue.md) — Single WebSocket reader + queue (deadlock fix)
- [ADR-0015](../adr/0015-browser-stt-quality.md) — Browser STT quality (short vs long phrases)
- [ADR-0016](../adr/0016-offline-model-ttft-long-utterances.md) — TTFT limitation of offline model
