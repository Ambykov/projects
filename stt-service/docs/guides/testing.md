# Тестирование — блоки A–H

## Окружение для тестов

Тесты запускаются только на Python 3.11:

```bash
python3.11 -m venv .venv-test
source .venv-test/bin/activate
pip install -r app/requirements-dev.txt
pytest tests/ -v
```

**Python 3.14 не поддерживается:** pytest-asyncio и FastAPI ещё не обновились под 3.14, warnings не критичны, но результаты могут расходиться с прод-окружением (Docker `python:3.11-slim`).

## Обзор блоков

| Блок | Описание | Критерий PASS |
|------|----------|---------------|
| A | Регресс-бенчмарк (144 файла) | WER 2.08% |
| B | Случайные тройки (20 троек) | ≥18/20, known limitations (иди, включи, вниз — ADR-0010) |
| C | Edge cases протокола | 8/8 |
| D | Auth-тесты | 7/7 |
| E4 | Backpressure (21-я сессия) | 4429 код |
| F | Метрики Prometheus | partials +2, final +3 |
| G | Bad audio тесты | 5/5 |
| H | Long utterance (4+ сек) | partials ≥2, TTFP сервер 800–2000 ms |

## Запуск всех тестов

```bash
python3 scripts/test_suite_stage4a.py
```

## Детали блоков

### Блок A: Регресс-бенчмарк

```bash
python3 scripts/test_suite_stage4a.py 2>&1 | grep "^\[A\]"
```

Ожидается:
- WER 2.08% (3 ошибки: иди→уйди, включи→ключи, вниз→низ)
- TTFP сервер: 100–300 ms (короткие), 800–2000 ms (длинные) — ADR-0016

### Блок C: Edge cases

```bash
python3 scripts/test_edge_cases.py
```

Ожидается:
- C1-C8 все PASS

### Блок D: Auth

```bash
python3 scripts/test_auth.py
```

Ожидается:
- D1-D7 все PASS
- D3: Authorization header работает
- D7: AUTH_MODE=none на порту 8101 (требуется `docker-compose.test.yml`)

### Блок E4: Backpressure

```bash
python3 scripts/test_backpressure.py
```

Ожидается:
- 21-я сессия получает код 4429

### Блок F: Метрики

```bash
python3 scripts/test_metrics_long.py
```

Ожидается:
- Длинная реплика: partials +2, TTFP +1
- Короткие реплики: partials +0, final +3

### Блок G: Bad audio

```bash
python3 scripts/test_bad_audio.py
```

Ожидается:
- G1-G5 все PASS (некорректное аудио отклоняется)

## Сырые логи

Сырые логи тестов сохраняются в `results/stage4a-tests/` (если папка создана).
