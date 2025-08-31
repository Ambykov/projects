# src/utils.py

import logging
import pandas as pd
import numpy as np
import Levenshtein as lev
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Функция простановки класса ---
def set_class(df, idx, code):
    current_class = df.at[idx, 'Класс']

    # Если текущий класс None или не список → ставим [code]
    if not isinstance(current_class, list):
        df.at[idx, 'Класс'] = [code]

    else:
        # Если есть 0 — заменяем его на code
        if 0 in current_class:
            new_class = [code if c == 0 else c for c in current_class]
        # Иначе добавляем, если ещё не было
        elif code not in current_class:
            new_class = current_class + [code]
        else:
            new_class = current_class

        df.at[idx, 'Класс'] = new_class

    return df


# --- Статистика по классам ---
def class_statistics(df):
    normalized = df['Класс'].apply(tuple)
    stats = normalized.value_counts().reset_index()
    stats.columns = ['Класс', 'Количество']
    stats['Класс'] = stats['Класс'].apply(lambda x: list(x))
    total = len(df)
    stats['Процент'] = (stats['Количество'] / total * 100).round(2)
    stats.loc[len(stats)] = ['Всего', total, 100.00]
    return stats.reset_index(drop=True)


# --- Токенизация текста ---
def tokenize(texts, tokenizer, max_length):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to('cpu')


# --- Разбиение на чанки ---
def split_text_into_chunks(text, tokenizer, max_length, stride):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        start += stride
    return chunks


# --- Очистка текстовых колонок ---
def clean_text_columns(df):
    def clean(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.strip()
        text = text.lower()
        # Замена спецсимволов между словами
        text = re.sub(r'(?<=[а-яА-Я])([^\w\s\/])(?=[а-яА-Я])', ' ', text)
        # Убираем повторяющиеся пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['Отзыв'] = df['Отзыв'].apply(clean)

    logger.info("✅ Текстовые колонки очищены")
    return df


# --- Проверка коротких отзывов (класс 101) ---
def is_short_symbol(text):
    if not isinstance(text, str) or not text.strip():
        return False

    text = text.strip()

    # Убираем знаки препинания между буквами
    normalized = re.sub(r'(?<=[а-яА-Я])([^\w\s\/])(?=[а-яА-Я])', ' ', text)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Проверяем длину
    if not (1 <= len(normalized) <= 3):
        return False

    if normalized.isspace():
        return False

    # Исключаем "бу", "БУ", "б/у" и т.д.
    cleaned_words = normalized.lower().replace(' ', '').replace('/', '')
    if cleaned_words == 'бу':
        return False

    return True


def is_fuzzy_match_with_details(review_text, key_phrases):
    """
    Проверяет, есть ли в тексте фразы с опечатками из key_phrases.
    Поддерживает:
        - список строк
        - список словарей с ключом 'phrases'

    Возвращает:
        bool: найдено ли хотя бы одно совпадение
        list: список кортежей (original, corrected, source)
    """
    matches = []

    review_words = review_text.split()

    for item in key_phrases:
        # Если это словарь с фразами
        if isinstance(item, dict):
            phrases = item.get("phrases", [])
        else:
            phrases = [item]

        for phrase in phrases:
            phrase_words = phrase.split()
            n = len(phrase_words)
            if n > len(review_words):
                continue

            # Поиск в тексте всех сегментов длины n
            for start_pos in range(len(review_words) - n + 1):
                segment = review_words[start_pos : start_pos + n]

                match = True
                original_words = []
                corrected_words = []

                for i, target_word in enumerate(phrase_words):
                    word = segment[i]
                    original_words.append(word)

                    # Полное совпадение
                    if word == target_word:
                        corrected_words.append(word)
                        continue

                    # Замена "ё" ↔ "е"
                    if word.replace("ё", "е") == target_word.replace("ё", "е"):
                        corrected_words.append(word)
                        continue

                    # Слова длиной 1–3 символа → точное совпадение или замена "не" ↔ "ни"
                    if len(target_word) <= 3:
                        if (target_word == "не" and word == "ни") or (target_word == "ни" and word == "не"):
                            corrected_words.append(word)
                        else:
                            match = False
                            break

                    # Слова 4–5 букв → Левенштейн ≤1
                    elif 4 <= len(target_word) <= 5:
                        if lev.distance(word, target_word) <= 1:
                            corrected_words.append(word)
                        else:
                            match = False
                            break

                    # Слова >5 букв → Левенштейн ≤2
                    else:
                        if lev.distance(word, target_word) <= 2:
                            corrected_words.append(word)
                        else:
                            match = False
                            break

                if match:
                    corrected_phrase = ' '.join(corrected_words)
                    phrase_joined = ' '.join(phrase_words)

                    # Добавляем совпадение (или с исправлением, или точное)
                    matches.append((
                        ' '.join(original_words),
                        corrected_phrase,
                        phrase_joined
                    ))
                    break  # после успешного совпадения этой фразы переходим к следующей

    return len(matches) > 0, matches




# --- Функция поиска близкого совпадения на уровне фразы ---
def is_fuzzy_match_phrase_level(review_words, key_phrases):
    """
    Проверяет, что отзыв полностью соответствует фразе, с допустимыми ошибками

    Для каждого слова:
        - 1–3 буквы → точное совпадение или замена "е"/"ё", "не"/"ни"
        - 4–5 букв → Левенштейн ≤1
        - >5 букв → Левенштейн ≤2

    Возвращает:
        bool: есть ли подходящая фраза
        list: список кортежей (original, corrected, source)
    """
    matches = []

    for item in key_phrases:
        # Если это словарь, ищем во фразах
        if isinstance(item, dict):
            for phrase in item.get("phrases", []):
                found, submatches = _check_phrase(review_words, phrase)
                if found:
                    matches.extend(submatches)
        else:
            # Если это просто строка
            found, submatches = _check_phrase(review_words, item)
            if found:
                matches.extend(submatches)

    return len(matches) > 0, matches


def _check_phrase(review_words, phrase):
    """Проверка одной фразы"""
    phrase_words = phrase.split()
    if abs(len(review_words) - len(phrase_words)) > 1:
        return False, []

    original_words = []
    corrected_words = []
    match = True

    for i, target_word in enumerate(phrase_words):
        if i >= len(review_words):
            return False, []

        word = review_words[i]
        original_words.append(word)

        # Полное совпадение
        if word == target_word:
            corrected_words.append(word)
            continue

        # Замена "ё" ↔ "е"
        if word.replace("ё", "е") == target_word.replace("ё", "е"):
            corrected_words.append(word)
            continue

        # Слова длиной 1–3 символа → точное совпадение или "не" ↔ "ни"
        if len(target_word) <= 3:
            if target_word == "не" and word == "ни":
                corrected_words.append(word)
            elif target_word == "ни" and word == "не":
                corrected_words.append(word)
            else:
                match = False
                break

        # Слова 4–5 букв → Левенштейн ≤1
        elif 4 <= len(target_word) <= 5:
            if lev.distance(word, target_word) <= 1:
                corrected_words.append(word)
            else:
                match = False
                break

        # Слова >5 букв → Левенштейн ≤2
        else:
            if lev.distance(word, target_word) <= 2:
                corrected_words.append(word)
            else:
                match = False
                break

    corrected_phrase = ' '.join(corrected_words)
    phrase_joined = ' '.join(phrase.split())

    if match and corrected_phrase != phrase_joined:
        return True, [(
            ' '.join(original_words),
            corrected_phrase,
            phrase_joined
        )]
    else:
        return False, []


##################################################################################
from decorators import pipeline_step
from config import NEGATIV_LEVEL, FILTERED_OUT_CLASS


@pipeline_step(step_number=2, step_name="ФИЛЬТРАЦИЯ ПО NEGATIV_LEVEL")
def filter_by_negativ_level(df, stop_pipeline_flag=False, negativ_level=None):
    """
    Фильтрует датафрейм: отзывы с оценкой > negativ_level не удаляются,
    а помечаются специальным классом FILTERED_OUT_CLASS.

    :param df: Входной датафрейм с колонкой 'Оценка'
    :param stop_pipeline_flag: Флаг прерывания шага
    :param negativ_level: Порог негативности
    :return: Обновлённый датафрейм
    """
    if stop_pipeline_flag:
        logger.warning("🔚 [!] Шаг [2] прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning("🟡 [2] Входной датафрейм пустой или None")
        return df

    if negativ_level is None:
        negativ_level = NEGATIV_LEVEL

    logger.info(f"🔍 [2] Применяю фильтр: оценки > {negativ_level} будут исключены из обработки (помечены классом {FILTERED_OUT_CLASS})")

    if 'Оценка' not in df.columns:
        logger.error("❌ [2] Колонка 'Оценка' отсутствует в датафрейме")
        raise ValueError("Колонка 'Оценка' обязательна для фильтрации")

    # Приводим оценку к числовому типу
    df['Оценка'] = pd.to_numeric(df['Оценка'], errors='coerce')

    # Отметим отзывы с оценкой > negativ_level специальным классом
    filtered_out_mask = df['Оценка'] > negativ_level
    filtered_out_indices = df[filtered_out_mask].index

    logger.info(f"🟡 [2] Отметка {len(filtered_out_indices)} отзывов как отфильтрованных")

    # Инициализируем колонку 'Класс', если её нет
    if 'Класс' not in df.columns:
        df['Класс'] = [[] for _ in range(len(df))]
    else:
        # Приводим всех к списку для единообразия
        def ensure_list(x):
            if isinstance(x, list):
                return x
            elif pd.isna(x):
                return []
            else:
                return [x]
        df['Класс'] = df['Класс'].apply(ensure_list)

    # Проставляем FILTERED_OUT_CLASS для отфильтрованных записей
    for idx in filtered_out_indices:
        df = set_class(df, idx, FILTERED_OUT_CLASS)

    # Возвращаем датафрейм без удаления строк — благодаря классам фильтр можно обходить
    logger.info(f"✅ [2] Фильтрация завершена: {len(filtered_out_indices)} отзывов помечено классом {FILTERED_OUT_CLASS}")
    return df



from decorators import pipeline_step
from config import logger
import re

@pipeline_step(step_number=7, step_name="ОЧИСТКА ТЕКСТА")
def step_clean_text(df, stop_pipeline_flag=False):
    import re
    logger.info("🧹 [7] ОЧИСТКА ТЕКСТА: старт")

    if stop_pipeline_flag:
        logger.warning("🔚 [!] Шаг очистки текста прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning("🟡 [7] Очистка текста пропущена — входной DataFrame пуст")
        return df  # Возвращаем df, а не None

    if 'Отзыв' not in df.columns:
        logger.error("❌ Нет колонки 'Отзыв' для очистки")
        return df  # Возвращаем df, а не None

    def clean(text):
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.strip().lower()

        # Заменяем все знаки препинания, спецсимволы и эмодзи на пробелы
        # Включаем в класс символов:
        # - все стандартные знаки пунктуации
        # - любые символы кроме букв, цифр, пробелов и /
        # Удаляем также эмодзи — класс символов из диапазонов Юникода
        # Для эмодзи можно использовать диапазоны Unicode, здесь общий пример

        # Unicode диапазоны для эмодзи, смайлов и др.
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # эмоции смайлы
            "\U0001F300-\U0001F5FF"  # символы и пиктограммы
            "\U0001F680-\U0001F6FF"  # транспорт и карты
            "\U0001F700-\U0001F77F"  # алхимические символы
            "\U0001F780-\U0001F7FF"  # геометрические символы
            "\U0001F800-\U0001F8FF"  # дополнительные символы и стрелки
            "\U0001F900-\U0001F9FF"  # дополнения к смайлам
            "\U0001FA00-\U0001FA6F"  # дополнения к символам
            "\U0001FA70-\U0001FAFF"
            "\U00002700-\U000027BF"  # различные символы
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(" ", text)

        # Заменяем все символы кроме букв, цифр, пробелов и / на пробел
        text = re.sub(r"[^a-zа-яё0-9\s/]", " ", text, flags=re.IGNORECASE)

        # Заменяем несколько пробелов на один и убираем пробелы в начале/конце
        text = re.sub(r"\s+", " ", text).strip()

        return text

    df['Отзыв'] = df['Отзыв'].apply(clean)

    logger.info("✅ [7] ОЧИСТКА ТЕКСТА: завершено")
    return df
