# Коды ошибок

## HTTP ошибки

| Код | Название | Описание | Действие клиента |
|-----|----------|----------|------------------|
| 400 | Bad Request | Неверный формат аудио (ожидался WAV) | Проверить формат файла, отправить WAV |
| 401 | Unauthorized | Отсутствует или неверен токен | Проверить токен, добавить `Authorization: Bearer <token>` |
| 413 | Payload Too Large | Файл превышает MAX_UPLOAD_MB | Уменьшить размер файла |
| 415 | Unsupported Media Type | Неподдерживаемый тип контента | Использовать `multipart/form-data` |
| 502 | Bad Gateway | Upstream GigaSTT недоступен | Повторить запрос позже, проверить `/ready` |

## WebSocket коды закрытия

| Код | Название | Описание | Действие клиента |
|-----|----------|----------|------------------|
| 4401 | Unauthorized | Неверный или отсутствующий токен | Проверить токен, переподключиться с правильным токеном |
| 4408 | Idle Timeout | Сессия простаивала > SESSION_IDLE_TIMEOUT_SEC | Переподключиться, отправить аудио быстрее |
| 4429 | Session Limit Exceeded | Достигнут MAX_CONCURRENT_SESSIONS | Подождать освобождения сессии или уменьшить нагрузку |
| 1000 | Normal Close | Нормальное закрытие (клиент отправил `stop`) | Завершить сессию |

## Примеры обработки ошибок

### Python (WebSocket)
```python
import websockets

try:
    async with websockets.connect(uri) as ws:
        await ws.send(audio_data)
except websockets.exceptions.ConnectionClosedError as e:
    if e.code == 4401:
        print("Ошибка аутентификации: проверьте токен")
    elif e.code == 4429:
        print("Достигнут лимит сессий: подождите")
```

### Python (REST)
```python
import httpx

response = httpx.post(url, files={"file": audio}, headers=headers)
if response.status_code == 400:
    print("Неверный формат аудио: используйте WAV")
elif response.status_code == 401:
    print("Ошибка аутентификации: проверьте токен")
elif response.status_code == 413:
    print("Файл слишком большой: уменьшите размер")
elif response.status_code == 502:
    print("GigaSTT недоступен: повторите позже")
```

## WebSocket error messages

В отличие от close codes, это сообщения внутри открытой сессии:

| Code | Описание | Частота |
|------|----------|---------|
| `upstream_ready_timeout` | GigaSTT не ответил на ready за 10s (нет свободного слота) | **основной** при нагрузке |
| `upstream_recv_timeout` | GigaSTT не ответил за 30s (зависание) | редкий |
| `upstream_connect_failed` | Не удалось подключиться к GigaSTT | редкий |
| `pool_unavailable` | `/ready` показал `pool_available=0` в момент проверки | **очень редкий** (race condition) |

**Для SDK:** `pool_unavailable` и `upstream_ready_timeout` — **оба означают «нет ёмкости»**, обрабатывайте одинаково (retry с backoff).

Сессия остаётся открытой, клиент может повторить.
