# Индекс архитектурных решений (ADR)

| # | Название | Статус | Дата |
|---|----------|--------|------|
| 0001 | Выбор STT: GigaSTT | Accepted | 2026-10-28 |
| 0002 | Протокол измерений | Accepted | 2026-10-28 |
| 0003 | CPU-only режим | Accepted | 2026-10-28 |
| 0004 | Формат аудио для тестов | Accepted | 2026-10-28 |
| 0005 | Partials для GigaSTT: длинные vs короткие реплики | Accepted | 2026-10-29 |
| 0006 | Per-utterance upstream reconnection | Accepted | 2026-10-29 |
| 0007 | Порты 8100/8101/8102 | Accepted | 2026-10-29 |
| 0008 | Docker verification A–H | Accepted | 2026-10-29 |
| 0009 | TTS архитектура — единый voice-gateway | Accepted | 2026-10-29 |
| 0010 | Исправление критерия блока B (статистическая корректность) | Accepted | 2026-10-29 |
| 0011 | Пустые final при прямом подключении — баг теста | Superseded | 2026-10-29 |
| 0012 | Gateway: single reader + asyncio.Queue | Accepted | 2026-10-29 |
| 0013 | diff opened/closed в логах gateway — артефакт логирования | Accepted | 2026-10-30 |
| 0015 | Качество STT в браузере — короткие vs длинные фразы | Accepted | 2026-10-30 |
| 0016 | TTFT растёт линейно с длиной фразы (ограничение offline-модели) | Accepted | 2026-10-30 |
| 0014 | Gateway error messages (Gap 1) | Accepted | 2026-10-30 |
| 0017 | Python SDK — одна реплика = одно WS-соединение (MVP) | Accepted | 2026-10-30 |
| 0018 | Результаты нагрузочных тестов (local baseline) | Accepted | 2026-11-02 |
| 0019 | TypeScript SDK v0.1.0 — фиксы и известные ограничения | Accepted | 2026-11-02 |
| 0020 | Voice-core — деплой на serverllm | Accepted | 2026-11-02 |
| 0021 | Нагрузочные тесты voice-core на serverllm | Accepted | 2026-11-02 |

## Шаблон для новых ADR

См. [template.md](template.md)

## Архив

Исторический файл [decisions.md](../decisions.md) сохранён для истории до 2026-10-29. Актуальные решения — в этом индексе.
