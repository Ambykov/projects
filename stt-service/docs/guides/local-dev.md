# Локальная разработка

## Архитектура локальной отладки

Для быстрой итерации при разработке шлюза:

- **gigastt** — в Docker контейнере (порт 9876 проброшен на хост)
- **stt-gateway** — локально через `uvicorn --reload` (порт 8100)

**Почему:** Пересборка Docker-контейнера занимает 2–3 минуты. `uvicorn --reload` позволяет видеть изменения за секунды.

## Настройка

### 1. Создать .env.local

```bash
# .env.local (для локального uvicorn, не коммитится)
GIGASTT_WS_URL=ws://127.0.0.1:9876/v1/ws
GIGASTT_REST_URL=http://127.0.0.1:9876/v1/transcribe
GIGASTT_HEALTH_URL=http://127.0.0.1:9876/health
GIGASTT_READY_URL=http://127.0.0.1:9876/ready
AUTH_MODE=api_key
API_KEY=testtoken
LOG_LEVEL=INFO
MAX_CONCURRENT_SESSIONS=20
MAX_UPLOAD_MB=10
SESSION_IDLE_TIMEOUT_SEC=120
CORS_ORIGINS=http://localhost:8100,http://127.0.0.1:8100
PUBLIC_HOSTNAME=localhost
```

**.env.local** — для локального uvicorn (не коммитится).
**.env** — для Docker Compose (см. `onboarding.md`).

### 2. Создать виртуальное окружение

```bash
python3.11 -m venv .venv-gateway
source .venv-gateway/bin/activate
pip install -r app/requirements-dev.txt
```

### 3. Запустить uvicorn

```bash
set -a
source .env.local
set +a
uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
```

## .gitignore

```
.env.local
.venv-gateway/
__pycache__/
*.pyc
```

## Полезные команды

```bash
# Проверить, что gigastt запущен
curl http://127.0.0.1:9876/health

# Проверить шлюз
curl http://127.0.0.1:8100/health

# Посмотреть логи gigastt
docker compose logs gigastt --tail 50

# Перезапустить gigastt
docker compose restart gigastt
```
