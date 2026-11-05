# Документация проекта ru-voice-assistant-bench

Проект голосового ассистента: STT (реализован), VAD (4.C), TTS (Stage 3, отложен), LLM (внешний, на сервере). Текущий фокус — STT как продукт.

---

## Быстрый старт

| Документ | Описание |
|----------|----------|
| [Onboarding](guides/onboarding.md) | Как войти в проект за 30 минут |
| [Локальная разработка](guides/local-dev.md) | uvicorn --reload + .env.local |
| [Тестирование](guides/testing.md) | Блоки A–H, как запускать |

---

## Архитектура

| Документ | Описание |
|----------|----------|
| [Обзор](architecture/overview.md) | Три слоя: модели / voice-gateway / SDK |
| [Поток данных](architecture/data-flow.md) | Микрофон → шлюз → GigaSTT → ответ |
| [Развёртывание](architecture/deployment.md) | Сервер, docker-compose, models_net |
| [Дорожная карта](architecture/roadmap.md) | Stage 1–6 |

---

## API

| Документ | Описание |
|----------|----------|
| [WebSocket](api/websocket.md) | Протокол: ready / partial / final / end_of_utterance / stop |
| [REST](api/rest.md) | /v1/transcribe, auth, лимиты |
| [Коды ошибок](api/errors.md) | 400 / 401 / 413 / 415 / 4429 + WS 4401 / 4408 / 4429 / 1000

---

## Решения (ADR)

| Документ | Описание |
|----------|----------|
| [Индекс ADR](adr/README.md) | Все архитектурные решения |
| [Шаблон ADR](adr/template.md) | Для новых ADR |

---

## Операции

| Документ | Описание |
|----------|----------|
| [Runbook](ops/runbook.md) | Что делать при инцидентах |
| [Мониторинг](ops/monitoring.md) | Метрики, Grafana, алерты |
| [Деплой](ops/deploy.md) | На сервер |

---

## Исследования

| Документ | Описание |
|----------|----------|
| [Stage 1: GigaSTT benchmark](research/stage1-gigastt-benchmark.md) | WER 2.08%, latency 226 мс |
| [Stage 2: T-one vs GigaSTT](research/stage2-tone-comparison.md) | Сравнение моделей |
| [Stage 3: TTS comparison](research/stage3-tts-comparison.md) | MOSS-TTS vs kokoro-ru — план (отложен до Stage 4) |

---

## Инциденты

| Документ | Описание |
|----------|----------|
| [2026-10-29 refactor regression](incidents/2026-10-29-refactor-regression.md) | Deadlock, 4408, race (Resolved) |

---

## Оперативное состояние

| Документ | Описание |
|----------|----------|
| [STATE.md](../STATE.md) | Текущая стадия, следующий шаг |
| [AGENTS.md](../AGENTS.md) | Инструкции для AI-агента |

---

## История

| Документ | Описание |
|----------|----------|
| [Changelog](changelog.md) | Версии и изменения |
| [decisions.md](decisions.md) | **Архив** ADR до 2026-10-29 |
| [api.md](api.md) | **Архив** API документации |
