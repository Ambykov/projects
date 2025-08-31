import os
import sys
import logging
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Добавляем корень проекта в путь
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Импорты HuggingFace
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

# Лемматизация
from pymorphy3 import MorphAnalyzer
import re

# Для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

# Логгирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classify_unlabeled")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(console_handler)

# Пути
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_CSV = os.path.join(RESULTS_DIR, f"unlabeled_classification_results_{TIMESTAMP}.csv")
RESULT_STATISTICS = os.path.join(RESULTS_DIR, f"statistics_{TIMESTAMP}.txt")
RESULT_PLOT = os.path.join(RESULTS_DIR, f"class_distribution_{TIMESTAMP}.png")

# Гиперпараметры
MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 42

# Фиксируем seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Инициализируем лемматизатор
morph = MorphAnalyzer()

def preprocess_text(text):
    """Приводит текст к нижнему регистру, удаляет пунктуацию, лемматизирует"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # удаление пунктуации
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens)

def load_unlabeled_data():
    logger.info("📥 Загрузка не размеченных данных из Excel...")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "format2_reviews.xlsx")

    # Загружаем данные
    df = pd.read_excel(DATA_PATH)

    # Объединяем поля
    df["Отзыв"] = df["Текст отзыва"].fillna("") + " " + df["Достоинства"].fillna("") + " " + df["Недостатки"].fillna("")
    df["Отзыв"] = df["Отзыв"].apply(preprocess_text)

    # Возвращаем только ID и текст
    return df[["ID отзыва", "Отзыв"]]

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        texts = [str(text) for text in examples["Отзыв"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    return dataset.map(tokenize_function, batched=True, remove_columns=["Отзыв"])

def classify_reviews(model, tokenizer, reviews_df):
    logger.info("📊 Классификация не размеченных отзывов...")

    # Создаем датасет
    tokenized_reviews = Dataset.from_pandas(reviews_df[['Отзыв']])
    tokenized_reviews = tokenized_reviews.map(
        lambda x: tokenizer(x['Отзыв'], padding="max_length", truncation=True, max_length=MAX_LENGTH),
        batched=True, remove_columns=["Отзыв"]
    ).with_format("torch")

    test_loader = DataLoader(tokenized_reviews, batch_size=BATCH_SIZE)

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    # Сохраняем результаты в DataFrame
    reviews_df["predicted_label"] = predictions
    reviews_df["prob_class_7"] = [p[1].item() for p in probabilities]
    reviews_df["prob_class_0"] = [p[0].item() for p in probabilities]

    return reviews_df

def save_results(df_results, result_csv):
    logger.info(f"📁 Сохранение результатов в {result_csv}...")
    df_results.to_csv(result_csv, index=False)
    logger.info(f"✅ Результаты сохранены в {result_csv}")

def print_statistics(df_results):
    total_reviews = len(df_results)
    class_7_count = (df_results["predicted_label"] == 1).sum()
    class_0_count = (df_results["predicted_label"] == 0).sum()

    class_7_percentage = (class_7_count / total_reviews) * 100
    class_0_percentage = (class_0_count / total_reviews) * 100

    logger.info("📊 Статистика классификации:")
    logger.info(f"Всего отзывов: {total_reviews}")
    logger.info(f"Класс 7: {class_7_count} ({class_7_percentage:.2f}%)")
    logger.info(f"Класс 0: {class_0_count} ({class_0_percentage:.2f}%)")

    # Сохраняем статистику в файл
    with open(RESULT_STATISTICS, "w", encoding="utf-8") as f:
        f.write(f"Дата и время: {TIMESTAMP}\n")
        f.write(f"Всего отзывов: {total_reviews}\n")
        f.write(f"Класс 7: {class_7_count} ({class_7_percentage:.2f}%)\n")
        f.write(f"Класс 0: {class_0_count} ({class_0_percentage:.2f}%)\n")

    return {
        "total_reviews": total_reviews,
        "class_7_count": class_7_count,
        "class_0_count": class_0_count,
        "class_7_percentage": class_7_percentage,
        "class_0_percentage": class_0_percentage
    }

def plot_distribution(df_results):
    logger.info("📉 Визуализация распределения классов...")

    plt.figure(figsize=(8, 5))
    df_results["predicted_label"].value_counts().plot(kind='bar', color=['#ff7f0e', '#1f77b4'])
    plt.title("Распределение классов среди не размеченных отзывов")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.xticks([0, 1], ["Класс 0", "Класс 7"], rotation=0)
    plt.tight_layout()
    plt.savefig(RESULT_PLOT)
    plt.close()
    logger.info(f"✅ График сохранён в {RESULT_PLOT}")
    return RESULT_PLOT

if __name__ == "__main__":
    logger.info("🚀 Начало классификации не размеченных данных")

    # Загружаем не размеченные данные
    df = load_unlabeled_data()

    # Путь к модели
    #ODEL_SAVE_DIR = r"./models/model_class_7_and_0_20250721_1127"
    MODEL_SAVE_DIR = r"./models/model_class_7_and_0_20250728_0702"
    logger.info(f"📥 Загрузка модели из {MODEL_SAVE_DIR}")

    # Загружаем модель и токенизатор
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR, num_labels=2)
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить модель: {e}")
        raise

    # Классифицируем отзывы
    classified_df = classify_reviews(model, tokenizer, df)

    # Сохраняем результаты
    save_results(classified_df, RESULT_CSV)

    # Выводим и сохраняем статистику
    stats = print_statistics(classified_df)

    # Строим и сохраняем график
    plot_distribution(classified_df)

    logger.info("✅ Классификация не размеченных отзывов завершена успешно!")