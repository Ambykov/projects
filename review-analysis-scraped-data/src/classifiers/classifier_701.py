# src/classifiers/classifier_701.py

import os
import re
import torch
import numpy as np
import pandas as pd
import pymorphy3
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import traceback

from decorators import pipeline_step
from config import (
    logger,
    WORD2IDX_701_PATH,
    EMBED_MATRIX_701_PATH,
    MODEL_701_WEIGHTS_PATH,
    SAVE_STEP_11_RESULT,
    PROCESSED_DIR,
    TIMESTAMP
)
from utils import set_class, class_statistics


MAX_LENGTH = 120
EMBEDDING_DIM = 300
HIDDEN_SIZE = 128
NUM_CLASSES = 2
CNN_KERNELS = (3, 4, 5)
CNN_FILTERS = 64
DROPOUT = 0.4


morph = pymorphy3.MorphAnalyzer()
pos_map = {
    'NOUN': 'NOUN', 'VERB': 'VERB', 'INFN': 'VERB', 'ADJF': 'ADJ', 'ADJS': 'ADJ',
    'COMP': 'ADJ', 'PRTF': 'ADJ', 'PRTS': 'ADJ', 'GRND': 'VERB', 'NUMR': 'NUM',
    'ADVB': 'ADV', 'NPRO': 'PRON', 'PRED': 'ADV', 'PREP': 'ADP', 'CONJ': 'CONJ',
    'PRCL': 'PART', 'INTJ': 'INTJ'
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
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(process_token(token) for token in tokens)


def text_to_sequence(text, word2idx, max_length=MAX_LENGTH):
    tokens = text.split()
    indices = [word2idx.get(t, 1) for t in tokens]  # 1 — <UNK>
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
                 kernel_sizes=(3, 4, 5), num_filters=64, dropout_rate=0.4, embed_matrix=None):
        super().__init__()
        if embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embed_matrix), freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_size,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_fc = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        convs = [torch.relu(conv(emb)) for conv in self.convs]
        min_len = min(conv.shape[2] for conv in convs)
        convs = [conv[:, :, :min_len] for conv in convs]
        cnn_out = torch.cat(convs, dim=1).transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        attended = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(attended)
        return self.fc(out)


def has_ignored_class(classes):
    if isinstance(classes, list):
        return 999 in classes or 100 in classes
    return classes == 999 or classes == 100


@pipeline_step(step_number=11, step_name="КЛАССИФИКАТОР [701]")
def classifier_701(df, stop_pipeline_flag=False, threshold=0.9, step_number=11):
    """
    Классификатор с порогом вероятности для положительного класса.

    :param df: DataFrame с колонкой 'Отзыв'
    :param stop_pipeline_flag: флаг остановки pipeline
    :param threshold: порог вероятности для классификации в класс 701
    :param step_number: номер шага пайплайна
    :return: DataFrame с добавленным классом 701 и вероятностями
    """
    logger.info(f"🧠 [{step_number}] КЛАССИФИКАТОР [701]: старт (порог={threshold})")
    try:
        if stop_pipeline_flag:
            logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
            return df

        if 'Отзыв' not in df.columns or 'Класс' not in df.columns:
            logger.error(f"❌ [{step_number}] Колонки 'Отзыв' или 'Класс' отсутствуют в датафрейме")
            return df

        # Фильтруем отзывы с классами 999 и 100
        df_filtered = df[~df['Класс'].apply(has_ignored_class)].copy()

        # Предобработка текста
        df_filtered['text_processed'] = df_filtered['Отзыв'].astype(str).apply(preproc_text)

        with open(WORD2IDX_701_PATH, "rb") as f:
            word2idx = pickle.load(f)
        embed_matrix = np.load(EMBED_MATRIX_701_PATH)

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
        model.load_state_dict(torch.load(MODEL_701_WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval()

        dataset = ReviewDataset(df_filtered['text_processed'].tolist(), word2idx)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        preds = []
        probs_701 = []

        for batch in tqdm(dataloader, desc=f"🧠 [{step_number}] Классификация [701]"):
            if stop_pipeline_flag:
                logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем внутри цикла")
                break
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)
            prob_class_701 = prob[:, 1]
            pred = (prob_class_701 > threshold).cpu().tolist()
            preds.extend(pred)
            probs_701.extend(prob_class_701.cpu().tolist())

        # Проставляем класс 701 для тех, у кого вероятность выше порога
        for idx, pred in zip(df_filtered.index, preds):
            if pred:
                df = set_class(df, idx, 701)

        # Добавляем вероятности класса 701 в основной df
        df.loc[df_filtered.index, 'prob_701'] = pd.Series(probs_701, index=df_filtered.index)

        # Удаляем временный столбец
        df_filtered.drop(columns=['text_processed'], inplace=True)
        if 'text_processed' in df.columns:
            df.drop(columns=['text_processed'], inplace=True)

        if SAVE_STEP_11_RESULT:
            try:
                output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_classifier_701_{TIMESTAMP}.xlsx")
                df.to_excel(output_file, index=False)
                logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
            except Exception as e:
                logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результаты classifier_701: {e}")

        try:
            stats_df = class_statistics(df)
            logger.info(f"\n📝 [{step_number}] Статистика классов после классификатора 701:\n%s", stats_df.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику после classifier_701: {e}")

        logger.info(f"✅ [{step_number}] КЛАССИФИКАТОР [701]: успешно выполнен")
        return df

    except Exception as ex:
        logger.error(f"❌ [{step_number}] КЛАССИФИКАТОР [701]: выполнение прервано из-за ошибки — {ex}")
        logger.error(traceback.format_exc())
        return None
