# 📁 Мои проекты

Репозиторий содержит мои основные ML- и Data-проекты, которые представлены в хронологическом порядке.


---

## 🏭 1. Создание ИИ-системы моделирования отказов оборудования на линии с экструдерами
- **Папка:** `/extruder-failure-prediction`
- **Описание:** Разработка ИИ-системы для прогнозирования тока экструдера на основе временных рядов. Цель — предотвращение отказов оборудования в промышленном производстве.
- **Стек:**: Python, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Keras, Matplotlib, pyarrow, psycopg2, socket, asyncio, time, datetime, openpyxl, python-docx, tqdm, logging
- **Ключевое:**
  - Гибридная модель с механизмом внимания
  - Интеграция с БД для потоковой обработки
  - Работающий пайплайн в реальном времени
  - Поддержка онлайн-прогнозирования
- **Ссылка**: https://ai-hunter.ru/amai

👉 [Перейти к проекту](extruder-failure-prediction/)

---

## 📊 2. ИИ-система по анализу и автоматизации обработки клиентских отзывов и вопросов на маркетплейсе с оценкой работы операторов
- **Папка:** `/review-analysis-client-data`
- **Описание:** Правиловая система автоматической классификации отзывов с Wildberries по 7 типам нарушений (спам, нецензурная лексика, политика и др.). Проект включает разработку словарей, исследование ML-моделей и создание устойчивого пайплайна.
- **Стек:** Python, Scikit-learn, PyTorch, Bert,  LSTM,  Gensim, pymorphy3, HuggingFace Transformers, time, datetime, openpyxl, python-docx, fuzzywuzzy, python-Levenshtein, tqdm, logging, loguru, colorlog, Matplotlib, Seaborn
- **Ключевое:**
  - Многопроходный пайплайн из 12 шагов
  - Поддержка прерывания и возобновления (`Ctrl+C`)
  - Гибридные классификаторы (правила + элементы ML)
  - Исследование моделей: `rubert-tiny2`, `CNN-BiLSTM + Attention`
  - Основа для автоматизированной системы мониторинга
- **Ссылка**: https://ai-hunter.ru/giper_april


👉 [Перейти к проекту](review-analysis-client-data/)

---

## 🕷️ 3. Pet-проект «Разработка системы автоматической классификации негативных отзывов на товары с использованием нейросетевых моделей»
- **Папка:** `/review-analysis-scraped-data`
- **Описание:** Автономный пайплайн, расширяющий `review-analysis-client-data` функционалом **сбора данных**. Включает парсинг артикулов и отзывов с Wildberries, после чего применяет уже отработанную систему классификации.
- **Стек:** Python, Scikit-learn, time, datetime, openpyxl, python-docx, fuzzywuzzy, python-Levenshtein, tqdm, logging, loguru, colorlog, BeautifulSoup, selenium, webdriver-manager, undetected-chromedriver.
- **Ключевое:**
  - Обход антибота через `undetected-chromedriver`
  - Поддержка возобновления парсинга и анализа
  - Использует **готовые словари и логику** из `review-analysis-client-data`
  - Автоматизация мониторинга репутации продавца
  - Генерация отчётов для подачи жалоб

> 🔗 **Важно:** Этот проект **основан на** `review-analysis-client-data`.
> Все классификаторы, словари и пайплайн анализа — те же, что и в проекте 2.
> ML-модели и словари **не переобучаются**, используются готовые.

- **Ссылка**: https://docs.google.com/presentation/d/1PIJWeQVomJzd6FSllHfsGrKqDK5CsafM9pdaRkghu1E/edit?usp=drive_link

👉 [Перейти к проекту](review-analysis-scraped-data/)

---

---

## 🎙️ 4. Голосовой сервис распознавания речи (STT): gateway + SDK

- **Папки:** `/stt-service` (gateway) + `/stt-sdk` (Python + TypeScript SDK)
- **Описание:** Продакшен-сервис распознавания речи (Speech-to-Text) для русского языка. WebSocket + REST API, авторизация по API-ключу, метрики Prometheus. Построен как sidecar над **GigaSTT** (GigaAM v3 RNNT, INT8, CPU-only). Клиентские SDK — Python (backend) и TypeScript (браузер, AudioWorklet).
- **Стек:** Python 3.11, FastAPI, uvicorn, websockets, httpx, Prometheus client, Docker, GigaSTT (Rust + ONNX Runtime), TypeScript, Vitest, AudioWorklet, pytest.
- **Ключевое:**
  - **Sidecar-паттерн:** модель `gigastt` в network namespace `stt-gateway` (loopback `127.0.0.1:9876`), снаружи недоступна — единая точка авторизации через gateway.
  - **WebSocket:** long-lived клиентское соединение, per-utterance upstream к модели (GigaAM v3 не держит long-lived WS).
  - **REST:** `POST /v1/transcribe` для batch-транскрипции.
  - **Auth:** `AUTH_MODE=api_key`, Bearer/query token, constant-time compare.
  - **Ошибки:** 4 in-session кода (`pool_unavailable`, `upstream_connect_failed`, `upstream_ready_timeout`, `upstream_recv_timeout`) + WS close codes (4401 / 4408 / 4429 / 1000).
  - **Метрики:** 62 series `stt_gateway_*` (Prometheus, VictoriaMetrics).
  - **SDK Python:** `STTClient.stream()` (one-connection-per-utterance), `WavAudioSource` / `MicAudioSource`, `RetryPolicy` (1s/2s/4s), typed `Timings`, 6 retryable-кодов. Python ≥ 3.11.
  - **SDK TypeScript:** `Session` (long-lived WS + AudioWorklet), `startUtterance`/`endUtterance` (streaming, без retry) и `sendUtterance(chunks)` (buffered, retry 3×), typed errors (`permission_denied`, `worklet_load_failed`, `connection_timeout`). Browser-only.
  - **Производительность:** WER 2.08% (MSC-ru), throughput 15.6–15.9 utt/s (pool=4), TTFT p50 ~240 мс на коротких фразах.
- **Известные ограничения:**
  - TTFT линейно растёт с длиной фразы (offline Conformer, ADR-0016). Для >5 сек — сегментация (VAD — planned).
  - Overlapping не поддерживается (Gap 2) — ждать `final` перед следующей репликой.
  - TS SDK не работает в Node.js (browser-only). Для backend — raw WebSocket (пример в `stt-sdk/typescript/examples/node/`).
  - Лицензия GigaAM — commercial use требует отдельного договора (см. `stt-service/README.md` §«Зависимости и лицензии upstream»).

👉 [stt-service/](stt-service/) — gateway (FastAPI, docker-compose, deploy)
👉 [stt-sdk/](stt-sdk/) — Python + TypeScript SDK


> 💡 Все проекты разработаны с учётом реальных требований: устойчивость к сбоям, логирование, читаемость кода и возможность интеграции в продакшен.
