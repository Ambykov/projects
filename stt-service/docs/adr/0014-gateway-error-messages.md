# ADR-0014: Структурированные ошибки в gateway (Gap 1)

**Статус:** Accepted (Gap 1 закрыт, Gap 2 — overlapping — отложен)
**Дата:** 2026-10-30

## Контекст

Gap 1 из `gateway-as-is.md`: клиент не получает `{"type": "error"}`, не может отличить `pool_unavailable` от `upstream_recv_timeout` от `idle_timeout`.

## Решение

Отправлять `{"type": "error", "code": ..., "message": ...}` при:
- `pool_unavailable` — `pool_available=0` в `/ready`
- `upstream_connect_failed` — `websockets.connect` упал
- `upstream_ready_timeout` — нет `ready` за 10s
- `upstream_recv_timeout` — нет `partial`/`final` за 30s

Сессия НЕ закрывается (кроме idle/limit/auth). Клиент может повторить.

## Последствия

- SDK обрабатывает `{"type": "error"}` с retry policy
- Gap 2 (overlapping) — отложен, документирован в `docs/api/websocket.md`

## Ссылки

- `docs/architecture/gateway-as-is.md` — Gap 1
- `docs/api/websocket.md` — раздел «Ошибки»

## Частота срабатывания (2026-10-30)

- `upstream_ready_timeout` — **основной** при нагрузке (3-я сессия из 3 при pool=2)
- `upstream_connect_failed` — при остановленном GigaSTT
- `upstream_recv_timeout` — при зависании GigaSTT
- `pool_unavailable` — **очень редкий** (race: `/ready` показывает 0 ровно в момент проверки). В реальном диалоге (с `end_of_utterance`) не срабатывает, потому что GigaSTT ставит в очередь, не reject.

**Для SDK:** `pool_unavailable` и `upstream_ready_timeout` — оба retry-able, обрабатывайте одинаково.
