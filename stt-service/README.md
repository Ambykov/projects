STT Service
Сервис распознавания речи (Speech-to-Text) для русского языка. WebSocket + REST API, авторизация по API-ключу, метрики Prometheus, structured logs.
Построен как sidecar над GigaSTT (GigaAM v3 RNNT, INT8). Модель работает на CPU, не требует GPU.
Что внутри
    • stt-gateway — FastAPI-шлюз: WebSocket-прокси, REST API, auth, backpressure, метрики.
    • gigastt — модель GigaSTT (sidecar: в network namespace stt-gateway, слушает 127.0.0.1:9876, снаружи недоступен).
    • stt-gateway-none (test only) — без auth для тестов, порт 8101.
Клиенты (SDK, приложения) обращаются только к stt-gateway:8000.
Быстрый старт
cp .env.example .env
# В .env: сгенерировать API_KEY
#   openssl rand -hex 32
# и подставить в API_KEY=<...>

docker compose up -d --build
Проверка:
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/version
Первый запуск: модель (~465 MB) скачивается при старте gigastt. При повторных запусках — кеш в volume.
Архитектура
stt-network (Docker bridge)
│
├── клиенты (SDK, приложения)
│     │ ws://stt-gateway:8000/ws/stt
│     ▼
└── namespace: stt-gateway
      ├── stt-gateway :8000
      │      │ ws://127.0.0.1:9876 (loopback)
      │      ▼
      └── gigastt   :9876 (loopback only)
Ключевое:
    • gigastt — в namespace stt-gateway (network_mode: service:stt-gateway). Снаружи недоступен.
    • Единственная точка авторизации — stt-gateway (AUTH_MODE=api_key).
    • Замена модели не затрагивает клиентов: меняется только gigastt в docker-compose.yml, DNS-имя stt-gateway:8000 и контракт WS/REST — те же.
Конфигурация
Все настройки — через .env (см. .env.example):
Переменная	Default	Описание
AUTH_MODE	api_key	api_key / jwt / none
API_KEY	—	≥32 hex (openssl rand -hex 32)
MAX_CONCURRENT_SESSIONS	20	Лимит WS-сессий
MAX_UPLOAD_MB	10	Лимит REST upload
SESSION_IDLE_TIMEOUT_SEC	30	Idle timeout WS (рекомендуется 120 в проде)
GIGASTT_WS_URL	ws://gigastt:9876/v1/ws	upstream WS
GIGASTT_REST_URL	http://gigastt:9876/v1/transcribe	upstream REST
GIGASTT_READY_URL	http://gigastt:9876/ready	проверка pool
PUBLIC_HOSTNAME	localhost	Возвращается в /version

API
    • WebSocket — docs/api/websocket.md (протокол, close codes, in-session errors, sequence diagram, TTFT).
    • REST — docs/api/rest.md (POST /v1/transcribe).
    • Коды ошибок — docs/api/errors.md.
Тесты
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d
python3 scripts/test_suite_stage4a.py
Метрики
Prometheus на /metrics (62 series): stt_gateway_ws_active_sessions, stt_gateway_ws_sessions_total, stt_gateway_ws_time_to_first_partial_seconds, stt_gateway_ws_time_to_final_seconds, stt_gateway_rest_requests_total, stt_gateway_upstream_errors_total.
SDK
Клиенты — отдельный репозиторий stt-sdk:
    • Python — pip install stt-client
    • TypeScript (браузер) — npm install stt-client
Лицензия
MIT — см. LICENSE.
Зависимости и лицензии upstream
    • GigaSTT (ghcr.io/ekhodzitsky/gigastt:2.22.0) — MIT, github.com/ekhodzitsky/gigastt.
    • GigaAM v3 — лицензия GigaAM, github.com/salute-developers/GigaAM. Commercial use — отдельный договор с правообладателем. Этот репозиторий не распространяет веса GigaAM.
Документация
    • docs/README.md — индекс.
    • docs/api/ — контракт WS + REST.
    • docs/architecture/ — overview, data-flow, gateway-as-is.
    • docs/guides/ — local-dev, testing, SDK-интеграция, STT → LLM.
    • docs/adr/ — архитектурные решения (ADR-0001 … 0019).
