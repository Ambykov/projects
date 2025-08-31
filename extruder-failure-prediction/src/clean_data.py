# clean_data.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_dataframe(df, value_col):
    """
    Чистит датафрейм:
    - Убирает дубли по source_time (берёт последнее значение)
    - Группирует по целым секундам
    """

    logger.debug(f"Очистка данных для {value_col}")

    if 'source_time' not in df.columns:
        logger.warning("source_time отсутствует в датафрейме")
        return df

    # Преобразуем время и убираем дубликаты по секундам
    df['source_time'] = pd.to_datetime(df['source_time']).dt.floor('s')
    df[value_col] = df[value_col].ffill().fillna(0)
    df.sort_values('source_time', inplace=True)
    df.drop_duplicates('source_time', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[['source_time', value_col]]