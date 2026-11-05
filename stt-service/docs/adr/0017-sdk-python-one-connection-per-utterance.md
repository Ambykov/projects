# ADR-0017: Python SDK — одна реплика = одно WS-соединение (MVP)

**Статус:** Accepted
**Дата:** 2026-10-30
**Связано:** ADR-0006 (per-utterance upstream), ADR-0014 (Gap 1)

## Контекст

Python SDK `stt-client` v0.1.0 реализует `stream()` как «открыть WS → отправить реплику → получить final → закрыть WS». Это расходится с ожиданием «long-lived client connection на всю сессию диалога», но протокол на проводе идентичен: `ready → audio → end_of_utterance → partial/final → stop`. Сервер видит корректную сессию, просто короткую.

Альтернатива — `STTSession` (long-lived API) — рассмотрена и отложена: +100–150 строк, +тесты, +доки, задержка запуска MVP.

## Решение

Для MVP Python SDK принимается **одна реплика = одно WS-соединение**. `stream()` остаётся основным API.

`STTSession` (long-lived) фиксируется как post-MVP (v0.2.0) по контракту:
- один WS на сессию, открывается в `__aenter__`, закрывается в `close()`/`__aexit__`;
- `send_utterance()` блокирует до `final`, повторный вызов при `is_busy` → `RuntimeError`;
- retry внутри — только retry-able коды;
- после 4401 — сессия закрыта без повтора.

## Последствия

**Положительные:**
- MVP запускается быстрее.
- Протокол на проводе корректен, сервер не требует изменений.
- Клиентский таймаут 30 с исключает idle-timeout 120 с (4408 не воспроизводится).

**Отрицательные (принимаем осознанно):**
- `stt_gateway_ws_sessions_total` = число реплик, не диалогов. В Grafana «сессии» = «реплики» — переименовать в документации или пометить.
- Load test 20 клиентов × 30 реплик: последовательно — ок; параллельно — упрётся в `MAX_CONCURRENT_SESSIONS=20`.
- Barge-in невозможен до v0.2.0.
- LLM-интеграция (Stage 6) потребует `dialogue_id` отдельно от WS-сессии.
- TypeScript SDK уже реализует long-lived (`session.ts`, `waitingFinal`). Расхождение между SDK — временное, выровнять в v0.2.0.

## Обязательные требования к v0.1.0 (до тега)

- [x] Обработка close codes 4401 / 4408 / 4429 как `STTError` с `code`/`retryable`.
- [x] `RetryPolicy` ретраит 4429/4408, не ретраит 4401.
- [ ] Минимум один integration-тест против живого gateway: `ready → partial → final`.
- [ ] `app/static/` закоммичен или откачен.
- [x] `sdk/python/stt_client.egg-info/` в `.gitignore` и удалён из git.
- [x] Тесты запускаются на Python 3.11 (не 3.14).

## TODO v0.2.0

- [ ] `STTSession` (long-lived) по контракту выше.
- [ ] TypeScript SDK: цикл retry с exponential backoff (сейчас только `isRetryable`-флаг).
- [ ] `upstream_recv_timeout`: из manual → mock или fake-upstream в integration-тесте.
- [ ] Barge-in (Gap 2 на сервере).

## Ссылки

- ADR-0006 — per-utterance upstream reconnection
- ADR-0014 — Gap 1 (structured errors)
- `docs/api/websocket.md` — контракт WS
- `sdk/typescript/src/session.ts` — long-lived референс
