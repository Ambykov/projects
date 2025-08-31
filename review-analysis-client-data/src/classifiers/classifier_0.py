# src/classifiers/classifier_0.py

import logging
import pandas as pd
from decorators import pipeline_step
from config import logger, NEGATIV_LEVEL, FILTERED_OUT_CLASS
from utils import set_class
import warnings

# Отключаем предупреждения от openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

logger = logging.getLogger('pipeline')

@pipeline_step(step_number=2, step_name="НАЧАЛЬНАЯ КЛАССИФИКАЦИЯ")
def classifier_0(df, stop_pipeline_flag=False, step_number=2):
    """
    Создаёт колонки 'Класс' и 'Примечание'.
    Проставляет:
      - класс 999 (FILTERED_OUT_CLASS) для отзывов с оценкой > NEGATIV_LEVEL,
      - класс 0 для остальных.
    """

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if df is None:
        logger.error(f"❌ [{step_number}] Входной датафрейм равен None")
        return df

    df = df.copy()

    # Приводим 'Оценка' к числовому типу для корректного сравнения
    if 'Оценка' in df.columns:
        df['Оценка'] = pd.to_numeric(df['Оценка'], errors='coerce')
    else:
        logger.warning(f"⚠️ [{step_number}] В датафрейме отсутствует колонка 'Оценка' — все отзывы помечены классом 0")
        df['Оценка'] = pd.NA

    # Инициализируем колонки
    # В 'Класс' — список с одним элементом
    def initial_class(row):
        if pd.isna(row['Оценка']):
            # Оценка отсутствует — считаем нейтральным (0)
            return [0]
        elif row['Оценка'] > NEGATIV_LEVEL:
            return [FILTERED_OUT_CLASS]
        else:
            return [0]

    df['Класс'] = df.apply(initial_class, axis=1)
    df['Примечание'] = ''

    # Добавим примечания к отфильтрованным
    filtered_mask = df['Класс'].apply(lambda cl: FILTERED_OUT_CLASS in cl)
    df.loc[filtered_mask, 'Примечание'] = "Отфильтрован по уровню оценки"

    logger.info(f"📌 [{step_number}] Начальная классификация завершена: {filtered_mask.sum()} отзывов с классом {FILTERED_OUT_CLASS}")

    return df
