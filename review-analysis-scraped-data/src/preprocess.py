# src/preprocess.py

import pandas as pd
import os
import logging
from data_loader import load_data
from config import logger, SAVE_STEP_1_RESULT, PROCESSED_DIR, TIMESTAMP
from decorators import pipeline_step


def preprocess_reviews(df):
    """
    Предобрабатывает датафрейм с отзывами.
    Работает как со старым, так и с новым форматом данных.
    """

    if df is None:
        logger.error("❌ [preprocess_reviews] Входной датафрейм равен None")
        return None

    df = df.copy()

    # --- Добавляем номер строки ---
    df['Номер строки'] = df.index + 1

    # --- Безопасная замена всех некорректных значений на пустую строку ---
    for col in ['Текст отзыва', 'Достоинства', 'Недостатки']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: '' if not isinstance(x, str) or pd.isna(x) else x
            )

    # --- Формируем общий отзыв ---
    df['Отзыв'] = (
        df['Текст отзыва'].fillna('') + " " +
        df['Достоинства'].fillna('') + " " +
        df['Недостатки'].fillna('')
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    # --- Продавец ---
    if 'Продавец' in df.columns:
        df['Продавец'] = df['Продавец'].fillna('неизвестный')
    else:
        df['Продавец'] = 'неизвестный'

    return df


@pipeline_step(step_number=1, step_name="ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА")
def preprocess_reviews_with_loading(input_df=None, stop_pipeline_flag=False, step_number=1):
    """
    Полный шаг [1]: загрузка и предварительная обработка данных.
    Вызывается из main.py через PIPELINE_STEPS
    """

    if stop_pipeline_flag:
        logger.warning(f"🔚 [INTERRUPT] Шаг [{step_number}] прерван пользователем")
        return None

    try:
        if input_df is None:
            df = load_data()
        else:
            df = input_df.copy()

        if df is None:
            raise ValueError(f"❌ [{step_number}] Входной датафрейм равен None")

        # --- Предобработка ---
        df_processed = preprocess_reviews(df)

        if df_processed is None:
            logger.error(f"❌ [{step_number}] Предобработка вернула None → прерываем шаг")
            return None

        # --- Сохранение результата ---
        from config import SAVE_STEP_1_RESULT, PROCESSED_DIR, TIMESTAMP

        if SAVE_STEP_1_RESULT:
            output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_final_{TIMESTAMP}.xlsx")
            df_processed.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")

        logger.info(f"✅ [{step_number}] Успешно обработано {len(df_processed)} строк")
        return df_processed

    except Exception as e:
        logger.error(f"❌ [{step_number}] Ошибка в preprocess_reviews_with_loading — {e}")
        return None
