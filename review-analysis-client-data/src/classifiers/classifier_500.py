# src/classifiers/classifier_500.py

import logging
import pandas as pd
import os
import re
import sys
import importlib.util
from tqdm import tqdm

from utils import set_class, class_statistics
from decorators import pipeline_step
from config import logger, SAVE_STEP_6_RESULT, PROCESSED_DIR, TIMESTAMP, FILTERED_OUT_CLASS, CLASS_100

logger = logging.getLogger('pipeline')


def load_keywords(file_path="key_words/keywords_500.py"):
    """
    Загружает KEY_WORDS_500 из внешнего файла.
    """
    if not os.path.isfile(file_path):
        logger.error(f"❌ Файл ключевых слов не найден: {file_path}")
        return []

    module_name = "keywords_500"
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        logger.error(f"❌ Не удалось создать модуль из {file_path}")
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'KEY_WORDS_500'):
            key_words = module.KEY_WORDS_500
            logger.info(f"✅ Загружено {len(key_words)} ключевых слов нецензурной лексики")
            return key_words
        else:
            logger.warning("❌ В модуле нет переменной KEY_WORDS_500")
            return []
    except Exception as e:
        logger.error(f"⚠️ Ошибка при загрузке модуля: {e}")
        return []


@pipeline_step(step_number=6, step_name="Поиск по ключевым словам нецензурной лексики")
def classifier_500(df, stop_pipeline_flag=False, step_number=6):
    """
    Присваивает класс [500], если отзыв содержит одно из ключевых слов.
    Игнорирует отзывы с классами [100] и [999].
    """

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг [500] прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной датафрейм пустой или None")
        return df

    # Проверка наличия колонок
    if 'Класс' not in df.columns:
        logger.error(f"❌ [{step_number}] Входной датафрейм не содержит колонку 'Класс'.")
        return df

    if 'Примечание' not in df.columns:
        df['Примечание'] = ''

    # Фильтрация отзывов с классами 999 и 100
    def has_ignored_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes or CLASS_100 in classes
        return classes == FILTERED_OUT_CLASS or classes == CLASS_100

    df_to_process = df[~df['Класс'].apply(has_ignored_class)].copy()

    from config import KEYWORDS_FILES
    keywords_path = KEYWORDS_FILES.get(500, "key_words/keywords_500.py")
    key_words = load_keywords(keywords_path)

    if not key_words:
        logger.warning(f"⚠️ [{step_number}] Нет ключевых слов → пропуск шага [6]")
        return df

    found_count = 0
    detected_reviews = []

    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"🔎 [{step_number}] Поиск слов [500]"):
        if stop_pipeline_flag:
            logger.warning(f"🔚 [{step_number}] Шаг прерван внутри цикла")
            break

        review_text = str(row['Отзыв']).lower()
        for word in key_words:
            if len(word) < 3:
                continue
            if re.search(rf'\b{re.escape(word)}\b', review_text, re.IGNORECASE):
                df = set_class(df, idx, code=500)

                old_note = df.at[idx, 'Примечание']
                new_note = f"{old_note} | найдено слово '{word}'" if old_note else f"найдено слово '{word}'"
                df.at[idx, 'Примечание'] = new_note

                found_count += 1

                detected_reviews.append({
                    'Номер строки': idx,
                    'Отзыв': df.at[idx, 'Отзыв'],
                    'Дата создания': str(df.at[idx, 'Дата создания']) if 'Дата создания' in df.columns else '',
                    'Продавец': df.at[idx, 'Продавец'] if 'Продавец' in df.columns else '',
                    'Ключевое слово': word
                })
                break  # Прекращаем поиск по словам для текущего отзыва после первого совпадения

    logger.info(f"📌 [{step_number}] Найдено {found_count} отзывов с ключевыми словами. Проставлен класс [500]")

    if SAVE_STEP_6_RESULT and found_count > 0:
        output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_keywords_{TIMESTAMP}.xlsx")

        # Убираем таймзоны из 'Дата создания'
        df_to_save = df.copy()

        if 'Дата создания' in df_to_save.columns:
            df_to_save['Дата создания'] = df_to_save['Дата создания'].apply(
                lambda x: x.replace(tzinfo=None) if pd.notna(x) and hasattr(x, 'tz') else x
            )

        try:
            df_to_save.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_500: {e}")

        details_file = os.path.join(PROCESSED_DIR, f"step_6_keywords_details_{TIMESTAMP}.xlsx")
        try:
            pd.DataFrame(detected_reviews).to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Детали найденных отзывов сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить детали classifier_500: {e}")

        try:
            stats = class_statistics(df)
            logger.info(f"\n📊 [{step_number}] Статистика по всем классам после classifier_500:\n")
            logger.info(stats.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику — {e}")

    return df
