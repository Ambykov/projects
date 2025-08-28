# db_connector.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
SERVER_AVAILABLE = True

def check_connection():
    """Проверка доступности сервера"""
    return SERVER_AVAILABLE


def fetch_tag_data(tag, start_time_str=None, end_time_str=None):
    """Эмуляция загрузки данных по тегу"""

    if not check_connection():
        logger.critical("Сервер недоступен. Остановка загрузки.")
        raise ConnectionError("Сервер базы данных недоступен.")

    if start_time_str is None or end_time_str is None:
        start_time_str = (datetime.now() - timedelta(seconds=CHUNK_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date_range = pd.date_range(start=start_time_str, end=end_time_str, freq=f'{CHUNK_SECONDS}s')
    df = pd.DataFrame({
        'source_time': date_range,
        f'value_{tag}': np.random.rand(len(date_range)) * 100
    })

    logger.debug(f"Загружены данные для тега {tag}: {len(df)} записей")
    return df