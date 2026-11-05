# 0012: Одиночный читатель WebSocket + asyncio.Queue в Gateway

**Статус:** Accepted

## Контекст

При одновременном старте 20 клиентов по 30 utterances каждый 3-8 клиентов получали `TimeoutError`. Корневая причина:

- `client_to_upstream` и `upstream_to_client` работали как concurrent tasks
- Обе задачи делили один WebSocket: `client_to_upstream` читал, `upstream_to_client` писал
- При получении final `upstream_to_client` возвращался, а `client_to_upstream` всё ещё блокировался на `websocket.receive()`
- Отмена задачи через `task_c2u.cancel()` не отменяла блокирующую операцию `receive()` cleanly
- Starlette оставлял WebSocket в некорректном состоянии, следующий `receive()` падал с `Cannot call "receive" once a disconnect message has been received`

Попытки «исправить» через `wait_for(receive, timeout=0.1)` или флаг `final_received` были плацебо — они создавали новые проблемы (таймауты отменяли `receive()`, polling был нестабильным).

## Решение

Рефакторинг архитектуры gateway:

1. **Одиночный читатель WebSocket** — только главный цикл в `websocket_stt` читает из WebSocket
2. **asyncio.Queue для аудио** — все аудио-сообщения кладутся в очередь
3. **`client_to_upstream` читает из очереди** — не из WebSocket, поэтому отмена работает cleanly

### Архитектура после рефакторинга

```
websocket_stt (главный цикл, единственный читатель websocket)
  │
  ├─ читает websocket.receive()
  ├─ classify: bytes → queue.put(audio); "end_of_utterance" → queue.put(None); "stop" → break
  │
  └─ для каждой реплики: process_utterance(queue)
       │
       ├─ client_to_upstream(queue): читает ИЗ ОЧЕРЕДИ, шлёт в upstream. Отменяем cleanly.
       ├─ upstream_to_client(): читает upstream, шлёт клиенту final/partial.
       └─ после final: task_c2u.cancel() — очередь пуста, отмена безопасна.
```

### Ключевые изменения в коде

1. `process_utterance` принимает `audio_queue: asyncio.Queue` вместо чтения из WebSocket
2. `client_to_upstream` читает `await audio_queue.get()` вместо `await websocket.receive()`
3. Главный цикл `websocket_stt` — единственное место, где вызывается `websocket.receive()`
4. При получении `end_of_utterance` в очередь кладётся sentinel `None`

## Альтернативы, рассмотренные и отвергнутые

1. **`wait_for(receive, timeout=0.1)` + polling через флаг `final_received`**
   - Отклонено: отмена `receive()` через timeout оставляет WebSocket в некорректном состоянии
   - Результат: `Cannot call "receive" once a disconnect message has been received`

2. **Увеличение pool_size GigaSTT**
   - Отклонено: проблема не в upstream, а в обработке WebSocket на стороне gateway

3. **Staggered старт клиентов (задержка 500ms)**
   - Отклонено: это workaround, не решение проблемы архитектуры

## Последствия

- **Положительные**: 10/10 прогонов по 20 клиентов × 30 utterances = 20/20 OK
- **Положительные**: 0 ошибок `Cannot call "receive" once a disconnect message has been received`
- **Положительные**: Отмена задач работает cleanly, нет утечек ресурсов
- **Нейтральные**: Небольшая задержка из-за очереди (менее 1мс на utterance)

## Статус проверки (2026-10-30, обновлено)

**Проверено:**
- 10 прогонов 20 клиентов × 30 utterances = 20/20 OK (после рефакторинга)
- 0 `Cannot call receive once disconnect` в логах
- Quality Gate A–H PASS
- Pool 2/2, opened == closed (5×3)

**Ранее известные проблемы — решены:**
- keepalive 1011 → таймауты upstream (15s/30s/5s)
- 4408 idle timeout → сброс `audio_queue`, `current_rep_task`
- `first_audio_chunk` race → `audio_queue.put` до `create_task`
- Контейнер в нерабочем состоянии → `finally: await upstream.close()` с timeout 5s

**Ссылка:** docs/incidents/2026-10-29-refactor-regression.md (Resolved)

## Статус

Блокер «race condition в gateway» снят. Quality Gate A–H PASS. Stage 4.B закрыт.
