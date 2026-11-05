# REST API

## Base URL

```
http://<host>:8100
```

## Аутентификация

```
Authorization: Bearer <API_KEY>
```

Или через query параметр:
```
/v1/transcribe?token=<API_KEY>
```

## Endpoints

### POST /v1/transcribe

Транскрибация аудиофайла.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): WAV аудиофайл

**Аудио требования:**
- Format: WAV (PCM16)
- Sample rate: 16 кГц (обязательно; 48 кГц не поддерживается — см. ADR-0004, формат аудио)
- Channels: mono
- Max size: 10 MB (настраивается через `MAX_UPLOAD_MB`)

**Response:**
```json
{
  "text": "расшифрованный текст",
  "duration_sec": 3.5,
  "latency_ms": 121.0
}
```

**Пример (curl):**
```bash
curl -X POST http://localhost:8100/v1/transcribe \
  -H "Authorization: Bearer testtoken" \
  -F "file=@audio.wav"
```

### GET /health

Liveness probe.

**Response:**
```json
{"status": "ok"}
```

### GET /ready

Readiness probe (проверка upstream GigaSTT).

**Response (ready):**
```json
{"status": "ready", "upstream": "gigastt"}
```

**Response (not ready):**
```json
{"status": "not_ready", "upstream": "unavailable"}
```

### GET /version

Версии компонентов.

**Response:**
```json
{
  "gateway": "0.1.0",
  "gigastt": "2.22.0",
  "public_hostname": "localhost"
}
```

### GET /metrics

Prometheus метрики для VictoriaMetrics.

**Content-Type:** `text/plain`

**Пример метрик:**
```
stt_gateway_ws_active_sessions 5
stt_gateway_ws_sessions_total{status="ok"} 120
stt_gateway_rest_requests_total{status="ok"} 45
stt_gateway_upstream_errors_total{type="ws"} 2
```

## Ошибки

| Код | Описание |
|-----|----------|
| 400 | Invalid audio format (ожидался WAV) |
| 401 | Missing or invalid authentication token |
| 413 | File too large (превышен MAX_UPLOAD_MB) |
| 502 | Upstream GigaSTT unavailable |

См. также [Коды ошибок](errors.md).
