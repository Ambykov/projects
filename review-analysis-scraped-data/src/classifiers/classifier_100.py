# src/classifiers/classifier_100.py

import logging
import pandas as pd
import re
import os
from tqdm import tqdm

from utils import set_class, class_statistics
from decorators import pipeline_step
from config import logger, SAVE_STEP_3_RESULT, PROCESSED_DIR, TIMESTAMP, FILTERED_OUT_CLASS, CLASS_100

logger = logging.getLogger('pipeline')


def is_empty_review(text):
    """Проверяет, является ли отзыв пустым или NaN"""
    if pd.isna(text) or not isinstance(text, str):
        return True
    text = text.strip()
    return len(text) == 0


def is_foreign_language(text):
    """Проверяет, написан ли отзыв на иностранном языке (нет кириллицы)"""
    if not isinstance(text, str) or not text.strip():
        return False

    cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
    return not bool(cyrillic_pattern.search(text))


def is_short_symbol(text):
    """Проверяет, является ли текст слишком коротким (1–3 символа), но не 'бу'"""
    if not isinstance(text, str) or not text.strip():
        return False

    text = text.strip()

    # Заменяем знаки препинания между буквами на пробел
    normalized = re.sub(r'(?<=[а-яА-Я])([^\w\s\/])(?=[а-яА-Я])', ' ', text)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    if not (1 <= len(normalized) <= 3):
        return False

    if normalized.isspace():
        return False

    cleaned_words = normalized.lower().replace(' ', '').replace('/', '')
    if cleaned_words == 'бу':
        return False

    return True


def _add_class_and_note(df, idx, class_code, note):
    """Добавляет класс и примечание в датафрейм, аккуратно дописывая примечание"""
    df = set_class(df, idx, class_code)
    old_note = df.at[idx, 'Примечание'] if 'Примечание' in df.columns else ''
    if old_note:
        new_note = f"{old_note}; {note}"
    else:
        new_note = note
    df.at[idx, 'Примечание'] = new_note
    return df


@pipeline_step(step_number=3, step_name="ПРОВЕРКА НА [100]: пустой, иностранный язык, короткий текст")
def classifier_100(df, stop_pipeline_flag=False, step_number=3):
    """
    Присваивает класс CLASS_100, если:
      - отзыв пустой
      - отзыв на иностранном языке
      - отзыв очень короткий (1–3 символа), но не "бу"

    Обрабатывает только отзывы, у которых нет фильтрующего класса FILTERED_OUT_CLASS.
    Причина пометки сохраняется в 'Примечание'.
    """

    if stop_pipeline_flag:
        logger.warning("🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной датафрейм пустой или None")
        return df

    # Проверяем, есть ли колонки, если нет - инициализируем
    if 'Класс' not in df.columns:
        df['Класс'] = [[0] for _ in range(len(df))]
    else:
        def ensure_list(x):
            if isinstance(x, list):
                return x
            elif pd.isna(x):
                return []
            else:
                return [x]
        df['Класс'] = df['Класс'].apply(ensure_list)

    if 'Примечание' not in df.columns:
        df['Примечание'] = ''

    # Исключаем отзывы с фильтрующим классом
    def has_filtered_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes
        return classes == FILTERED_OUT_CLASS

    df_filtered = df[~df['Класс'].apply(has_filtered_class)]

    empty_count = 0
    foreign_count = 0
    short_count = 0

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="🔍 Проверка на [100]"):
        review_text = row.get('Отзыв', '')

        if is_empty_review(review_text):
            df = _add_class_and_note(df, idx, CLASS_100, 'пустой отзыв')
            empty_count += 1
            continue

        if is_foreign_language(review_text):
            df = _add_class_and_note(df, idx, CLASS_100, 'иностранный язык')
            foreign_count += 1
            continue

        if is_short_symbol(review_text):
            df = _add_class_and_note(df, idx, CLASS_100, 'очень короткий текст')
            short_count += 1

    logger.info(f"📊 [{step_number}] Найдено {empty_count} пустых отзывов. Проставлен класс [{CLASS_100}]")
    logger.info(f"🌐 [{step_number}] Найдено {foreign_count} отзывов на иностранном языке. Проставлен класс [{CLASS_100}]")
    logger.info(f"✂️  [{step_number}] Найдено {short_count} очень коротких отзывов. Проставлен класс [{CLASS_100}]")

    if SAVE_STEP_3_RESULT:
        try:
            output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_marked_{TIMESTAMP}.xlsx")
            df.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_100: {e}")

    # --- Выводим статистику ---
    try:
        stats = class_statistics(df)
        logger.info(f"\n📊 [{step_number}] Статистика по всем классам после classifier_100:\n")
        logger.info(stats.to_string(index=False))
    except Exception as e:
        logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику — {e}")

    return df
