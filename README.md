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

> 💡 Все проекты разработаны с учётом реальных требований: устойчивость к сбоям, логирование, читаемость кода и возможность интеграции в продакшен.