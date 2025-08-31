# 🕷️ Папка: `scraper/`

> **Автоматический парсинг данных с Wildberries**

Эта папка содержит скрипты для **сбора данных с платформы Wildberries**, включая:
- Парсинг **всех артикулов продавца**
- Парсинг **отзывов по каждому артикулу**

Данные используются в пайплайне анализа как **входной сырой датасет**.

---

## 📁 Структура папки
scraper/
├── parse_seller_products.py # Парсинг списка артикулов по ID продавца
├── reviews_parser.py # Парсинг отзывов по одному артикулу
├── logs/ # Логи выполнения
├── results/
| ├── reviews/ # Результаты парсинга отзывов
│ └── sellers/ # Результаты парсинга артикулов
├── requirements.txt # Зависимости
└── README.md # Этот файл


---

## 🧩 Основные скрипты

### 1. `parse_seller_products.py`

**Назначение**:
Парсинг **всех артикулов продавца** с Wildberries по его `seller_id`.

**Собираемые данные**:
- `ID продавца`
- `Название продавца`
- `артикул` (nm_id)
- `название` товара
- `ссылка`
- `дата парсинга`

**Особенности**:
- Обход антибота через `undetected-chromedriver`
- Прокрутка страницы для полной загрузки
- Перезапуск браузера каждые 5 страниц
- Автоматическое принятие кук
- Сохранение в `results/sellers/`

**Запуск**:
```
python parse_seller_products.py
# → Введите ID продавца: 250000402
```

## 2. reviews_parser.py
Назначение:
Парсинг отзывов по одному артикулу.

**Собираемые данные**:

покупатель, дата, оценка
достоинства, недостатки, комментарий
категория_уровень_1/2/3, бренд
Преобразование дат: "сегодня", "вчера", "5 мая 2025"

**Особенности**:
Гибкая обработка дат
Поддержка прокрутки до 3 минут
Возвращает pandas.DataFrame
Логирование в logs/

**Пример использования**:
```
from reviews_parser import reviews_parser
df = reviews_parser(12345678)
```

## 📦 Зависимости
```
selenium
undetected-chromedriver
pandas
loguru
tqdm
openpyxl

```
## Установка:
```
pip install -r requirements.txt
```
🔹 Требуется Google Chrome (версия 138+)


# --- Логи и результаты ---
logs/
results/
*.log
*.tmp

# --- Временные файлы браузера ---
webdriver/
chromedriver*

## 🚀 Как использовать
# 1. Парсинг всех артикулов продавца
```
python parse_seller_products.py
→ Введите ID продавца (например, 250000402)
```

# 2. Парсинг отзывов по списку артикулов
```
import pandas as pd
from reviews_parser import reviews_parser

df_skus = pd.read_excel("results/sellers/wb_seller_250000402_20250620.xlsx")
all_reviews = []

for sku in df_skus['артикул']:
    df = reviews_parser(sku)
    if not df.empty:
        all_reviews.append(df)
    time.sleep(random.uniform(10, 30))

final_df = pd.concat(all_reviews, ignore_index=True)
final_df.to_excel("all_reviews.xlsx", index=False)



📝 Примечание
Скрипты интегрированы в проект review-analysis-scraped-data и служат источником данных для пайплайна анализа.
