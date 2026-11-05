# ADR-0005: Partials для GigaSTT — длинные vs короткие реплики

**Статус:** Accepted
**Дата:** 2026-10-29
**Переименовано:** 2026-10-29 (объединены дубликаты 0005-gigastt-no-partial-for-short-utterances.md и 0005-gigastt-partials-поведение.md)

## Контекст

При тестировании Stage 4.A (Блок F и H, Docker verification) обнаружено, что метрика `stt_gateway_ws_time_to_first_partial_seconds` не увеличивается для коротких реплик, но работает для длинных.

GigaSTT модель (gigaam-v3-rnnt v2.22.0) имеет следующее поведение:

| Длительность utterance | Partial | TTFP |
|------------------------|---------|------|
| < 1 сек | ❌ Нет | N/A |
| 1–2 сек | ⚠️ Иногда | ~100–200ms |
| > 2 сек | ✅ Да | ~121ms (median) |

## Подтверждённые данные (Блок H, Docker)

**4-секундная реплика (склейка 4 файлов):**
```
PARTIAL @ 121ms: "стой"
PARTIAL @ 356ms: "иди"
FINAL @ 675ms: "Стой, иди стой!"

Total partials: 2
TTFP: 121ms
```

**Короткие реплики (~1 сек):**
```
FINAL @ +3ms: "фразы"

Total partials: 0
TTFP: N/A
```

## Решение

1. **SDK не полагается на partial для коротких команд (<1 сек)** — сразу ждёт final.
2. **Для длинных фраз partial работают** — TTFP 107–121 мс.
3. **Synthetic partial на стороне шлюза — отклонено** — искажает реальную latency.

## Поведение шлюза

- При получении первого partial: `partials_received_total.inc()` + `ws_time_to_first_partial.observe(ttfp)`
- При получении subsequent partial: только `partials_received_total.inc()`
- Если partial не приходит (короткая реплика): только `ws_time_to_final.observe(ttft)`

## Влияние на метрики

| Сценарий | partials_received_total | ws_time_to_first_partial_seconds_count | ws_time_to_final_seconds_count |
|----------|-------------------------|----------------------------------------|--------------------------------|
| Короткая реплика (< 1 сек) | +0 | +0 | +1 |
| Длинная реплика (> 2 сек, 2 partials) | +2 | +2 | +1 |

**В Grafana:** `partials_received_total` растёт медленнее, чем `ws_time_to_final_seconds_count` — это нормально.

## Последствия

1. SDK должно ресемплировать аудио в 16 kHz перед отправкой (GigaSTT не ресемплирует автоматически).
2. Для коротких команд SDK не должно ожидать partial — сразу ждать final.
3. TTFP ~107–121ms для длинных реплик — это нормально.
4. Метрика `partials_received_total` может расти медленнее, чем `ws_time_to_final_seconds_count` на длинных utterance'ах.

## Примечания

- GigaSTT версия: 2.22.0
- Модель: gigaam-v3-rnnt
- Тестирование: Блок F и H, Docker verification, 2026-10-29
