# в этом скрипте в файл пишется исходный текст отзыва, а не лемматизированный текст отзыва

import os
import re
import logging
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pymorphy3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Пути и параметры
PROJECT_ROOT = r"C:\Users\Андрей\Documents\Курсы\AI_ML_разработчик\Стажировка_2\Ноутбуки\Пайплайн"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "format2_reviews.xlsx")  # актуальный путь к файлу с отзывами

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "cnn_bilstm_attention_unlabeled")
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_CSV = os.path.join(RESULTS_DIR, f"unlabeled_predictions_{TIMESTAMP}.csv")
STATS_PATH = os.path.join(RESULTS_DIR, f"unlabeled_stats_{TIMESTAMP}.txt")
PLOT_PATH = os.path.join(RESULTS_DIR, f"class_distribution_{TIMESTAMP}.png")

#WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_21072025_1941", "word2idx.pkl")
#EMBED_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_21072025_1941", "embed_matrix.npy")
#MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_21072025_1941", "model_20250721_1941.pt")

#WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_26072025_1857", "word2idx.pkl")
#EMBED_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_26072025_1857", "embed_matrix.npy")
#MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_26072025_1857", "model_20250726_1857.pt")

# длина 120, словарь 15 000
#WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1031", "word2idx.pkl")
#EMBED_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1031", "embed_matrix.npy")
#MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1031", "model_20250727_1031.pt")

# длина 258, словарь 15 000
#WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1059", "word2idx.pkl")
#EMBED_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1059", "embed_matrix.npy")
#MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1059", "model_20250727_1059.pt")

# длина 258, словарь 25 000
WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1216", "word2idx.pkl")
EMBED_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1216", "embed_matrix.npy")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_bilstm_attention_27072025_1216", "model_20250727_1216.pt")

MAX_LENGTH = 258
EMBEDDING_DIM = 300
HIDDEN_SIZE = 128
NUM_CLASSES = 2
CNN_KERNELS = (3, 4, 5)
CNN_FILTERS = 64
BATCH_SIZE = 32
DROPOUT = 0.4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNN_BiLSTM_Unlabeled_Test")

morph = pymorphy3.MorphAnalyzer()
pos_map = {
    'NOUN': 'NOUN','VERB': 'VERB','INFN': 'VERB','ADJF': 'ADJ','ADJS': 'ADJ',
    'COMP': 'ADJ','PRTF': 'ADJ','PRTS': 'ADJ','GRND': 'VERB','NUMR': 'NUM',
    'ADVB': 'ADV','NPRO': 'PRON','PRED': 'ADV','PREP': 'ADP','CONJ': 'CONJ',
    'PRCL': 'PART','INTJ': 'INTJ'
}

def get_pos_tag(word):
    parse = morph.parse(word)[0]
    pos = parse.tag.POS
    return pos_map.get(pos, 'X')

def process_token(token):
    parse = morph.parse(token)[0]
    lemma = parse.normal_form
    pos = get_pos_tag(token)
    return f"{lemma}_{pos}"

def preproc_text(text):
    tokens = re.findall(r'\w+', text.lower())
    return " ".join(process_token(token) for token in tokens)

def text_to_sequence(text, word2idx, max_length=MAX_LENGTH):
    tokens = text.split()
    indices = [word2idx.get(t, 1) for t in tokens]  # 1 == <UNK>
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

class ReviewDataset(Dataset):
    def __init__(self, texts, word2idx):
        self.sequences = [text_to_sequence(t, word2idx) for t in texts]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 kernel_sizes=(3,4,5), num_filters=64, dropout_rate=0.4, embed_matrix=None):
        super().__init__()
        if embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embed_matrix), freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_fc = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        emb = self.embedding(x).transpose(1,2)
        convs = [torch.relu(conv(emb)) for conv in self.convs]
        min_len = min(conv.shape[2] for conv in convs)
        convs = [conv[:, :, :min_len] for conv in convs]
        cnn_out = torch.cat(convs, dim=1).transpose(1,2)
        lstm_out, _ = self.lstm(cnn_out)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        attended = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(attended)
        return self.fc(out)

def classify_unlabeled(model, dataloader, device):
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            probs.extend(p.cpu().tolist())
    return preds, probs

def save_results(df, preds, probs, path):
    df_copy = df.copy()
    df_copy['predicted_label'] = preds
    df_copy['prob_class_0'] = [p[0] for p in probs]
    df_copy['prob_class_7'] = [p[1] for p in probs]
    # Записываем исходный необработанный текст (из 'text_raw') без лемматизации
    df_copy.to_csv(path, index=False)
    logger.info(f"Результаты сохранены в {path}")

def save_stats(preds, path, timestamp):
    total = len(preds)
    count_7 = sum(1 for p in preds if p == 1)
    count_0 = total - count_7
    perc_7 = count_7 / total * 100 if total > 0 else 0
    perc_0 = 100 - perc_7

    logger.info(f"Всего отзывов: {total}")
    logger.info(f"Класс 7: {count_7} ({perc_7:.2f}%)")
    logger.info(f"Класс 0: {count_0} ({perc_0:.2f}%)")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Всего отзывов: {total}\n")
        f.write(f"Класс 7: {count_7} ({perc_7:.2f}%)\n")
        f.write(f"Класс 0: {count_0} ({perc_0:.2f}%)\n")

def plot_distribution(preds, path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(7,5))
    sns.countplot(x=preds)
    plt.xticks([0,1], ['Класс 0', 'Класс 7'])
    plt.title("Распределение предсказанных классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"График распределения классов сохранён в {path}")

def main():
    logger.info("Загрузка неразмеченных данных...")
    df = pd.read_excel(DATA_PATH)
    # Формируем колонку с исходным необработанным текстом
    df['text_raw'] = df['Текст отзыва'].fillna('') + ' ' + df['Достоинства'].fillna('') + ' ' + df['Недостатки'].fillna('')
    # Обрабатываем для модели (лемматизация + POS)
    df['text'] = df['text_raw'].apply(preproc_text)

    logger.info("Загрузка ресурсов модели...")
    with open(WORD2IDX_PATH, "rb") as f:
        word2idx = pickle.load(f)
    embed_matrix = np.load(EMBED_MATRIX_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_BiLSTM_Attention(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        kernel_sizes=CNN_KERNELS,
        num_filters=CNN_FILTERS,
        dropout_rate=DROPOUT,
        embed_matrix=embed_matrix
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)

    dataset = ReviewDataset(df['text'].tolist(), word2idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Классификация...")
    preds, probs = classify_unlabeled(model, dataloader, device)

    # Сохраняем результаты, используя исходный необработанный текст в df['text_raw']
    save_results(df[['ID отзыва', 'text_raw']], preds, probs, RESULT_CSV)
    save_stats(preds, STATS_PATH, TIMESTAMP)
    plot_distribution(preds, PLOT_PATH)

    logger.info("Классификация завершена")

if __name__ == "__main__":
    main()
