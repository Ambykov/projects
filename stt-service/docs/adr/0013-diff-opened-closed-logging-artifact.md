# ADR-0013: diff opened/closed в логах gateway — артефакт логирования, не утечка

**Статус:** Accepted
**Дата:** 2026-10-30

## Контекст

После серии тестов (3×20×30, 1800 utterances) в логах gateway:
- `Upstream opened`: 1809
- `Upstream closed`: 1806
- diff = 3

При этом `pool_available: 2/2` — все слоты пула gigastt свободны.

## Причина

`Upstream opened` логируется **после** `await websockets.connect(...)` — корректно. Но `Upstream closed` логируется **только** при нормальном закрытии в `finally`. Если:
- `gigastt` перезапускается (`docker compose restart gigastt`), старые upstream-соединения умирают на стороне gigastt;
- gateway получает `ConnectionClosed`, но `finally: await upstream.close()` на уже мёртвом соединении ничего не делает, и `logger.info("Upstream closed")` не вызывается.

Отсюда diff = 3 — это 3 upstream, которые были открыты до restart gigastt и умерли вместе с ним. Слоты пула при этом освобождаются на стороне gigastt (`pool_available=2`).

## Решение

Зафиксировать как known artifact. `diff > 0` в логах **не означает** утечку слотов — проверять `pool_available` через `/ready`. Если `pool_available == pool_total` — утечки нет.

## Последствия

- Мониторинг: алерт на `pool_available < pool_total` (реальная утечка), не на `diff opened/closed` (артефакт).
- Для прода: после restart gigastt возможен временный `diff > 0`, не требует вмешательства.
