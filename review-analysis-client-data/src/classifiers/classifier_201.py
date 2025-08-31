# src/classifiers/classifier_201.py

import logging
import pandas as pd
import os
import re
import sys
import importlib.util
from tqdm import tqdm

from utils import set_class, class_statistics
from decorators import pipeline_step
from config import logger, SAVE_STEP_8_RESULT, PROCESSED_DIR, TIMESTAMP, FILTERED_OUT_CLASS, CLASS_100, KEYWORDS_FILES

logger = logging.getLogger('pipeline')


def load_keywords_201():
    """
    Загружает KEY_WORDS_201 из файла по пути из конфига KEYWORDS_FILES[201].
    """
    file_path = KEYWORDS_FILES.get(201, "key_words/key_words_201.py")
    if not os.path.isfile(file_path):
        logger.error(f"❌ Файл ключевых слов не найден: {file_path}")
        return []

    module_name = "keywords_201"
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        logger.error(f"❌ Не удалось создать модуль из {file_path}")
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'KEY_WORDS_201'):
            key_words = module.KEY_WORDS_201
            # Компилируем паттерны с учётом word boundaries и игнорированием регистра
            compiled_patterns = [
                (word, re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE))
                for word in key_words
            ]
            logger.info(f"✅ Загружено и скомпилировано {len(key_words)} ключевых фраз для класса 201")
            return compiled_patterns
        else:
            logger.warning("❌ В модуле нет переменной KEY_WORDS_201")
            return []
    except Exception as e:
        logger.error(f"⚠️ Ошибка при загрузке модуля: {e}")
        return []


@pipeline_step(step_number=8, step_name="ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ [201] - Б/У")
def classifier_201(df, stop_pipeline_flag=False, step_number=8):
    """
    Присваивает класс [201], если отзыв содержит ключевые слова/фразы,
    указывающие на то, что товар был в употреблении (б/у).
    Игнорирует отзывы с классами [100], [999].
    """

    logger.info(f"🔍 [{step_number}] КЛАССИФИКАТОР [201]: старт – поиск признаков б/у товаров")

    if stop_pipeline_flag:
        logger.warning(f"🔚 {step_number}] Шаг [201] прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной датафрейм пустой или None")
        return df

    if 'Класс' not in df.columns:
        logger.error(f"❌ [{step_number}] Входной датафрейм не содержит колонку 'Класс'.")
        return df

    if 'Примечание' not in df.columns:
        df['Примечание'] = ''

    def has_ignored_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes or CLASS_100 in classes
        return classes == FILTERED_OUT_CLASS or classes == CLASS_100

    df_to_process = df[~df['Класс'].apply(has_ignored_class)].copy()

    compiled_patterns = load_keywords_201()

    if not compiled_patterns:
        logger.warning(f"⚠️[{step_number}] Нет ключевых слов для класса [201] → пропуск шага")
        return df

    found_count = 0
    detected_reviews = []

    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="🔎 Поиск б/у слов [201]"):
        if stop_pipeline_flag:
            logger.warning(f"🔚 [{step_number}] Шаг прерван внутри цикла")
            break

        review_text = str(row.get('Отзыв', '')).lower()

        found_words_set = set()  # чтобы избежать повторов

        for word, pattern in compiled_patterns:
            matches = pattern.findall(review_text)
            if matches:
                found_words_set.add(word)

        if found_words_set:
            df = set_class(df, idx, code=201)
            found_count += 1

            old_note = df.at[idx, 'Примечание']
            found_str = ", ".join(sorted(found_words_set))
            new_note = f"{old_note} | б/у признак: '{found_str}'" if old_note else f"б/у признак: '{found_str}'"
            df.at[idx, 'Примечание'] = new_note

            detected_reviews.append({
                'Номер строки': idx,
                'Отзыв': df.at[idx, 'Отзыв'],
                'Дата создания': str(df.at[idx, 'Дата создания']) if 'Дата создания' in df.columns else '',
                'Продавец': df.at[idx, 'Продавец'] if 'Продавец' in df.columns else '',
                'Найденные признаки б/у': found_str
            })

    logger.info(f"📌 [{step_number}] Найдено {found_count} отзывов с признаками б/у товаров. Проставлен класс [201]")

    if SAVE_STEP_8_RESULT and found_count > 0:
        output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_used_items_{TIMESTAMP}.xlsx")

        df_to_save = df.copy()
        if 'Дата создания' in df_to_save.columns:
            df_to_save['Дата создания'] = df_to_save['Дата создания'].apply(
                lambda x: x.replace(tzinfo=None) if pd.notna(x) and hasattr(x, 'tz') else x
            )

        try:
            df_to_save.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_201: {e}")

        details_file = os.path.join(PROCESSED_DIR, f"step_8_used_items_details_{TIMESTAMP}.xlsx")
        try:
            pd.DataFrame(detected_reviews).to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Детали найденных отзывов сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️[{step_number}] Не удалось сохранить детали classifier_201: {e}")

        try:
            stats = class_statistics(df)
            logger.info(f"\n📊 [{step_number}] Статистика по всем классам после classifier_201:\n")
            logger.info(stats.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику — {e}")

    return df
