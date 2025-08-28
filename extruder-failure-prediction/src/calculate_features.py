# calculate_features.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# calculate_features.py (версия для реального времени)
import pandas as pd
import numpy as np
import logging
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class FeatureCalculator:
    def __init__(self, max_history: int = 100):
        """Калькулятор фичей с сохранением истории для реального времени."""
        self.history: Dict[str, deque] = {}
        self.max_history = max_history

    def calculate_features(self, filtered_df: pd.DataFrame,
                          value_col: str,
                          num_lag: int = 6) -> pd.DataFrame:
        """
        Рассчитывает фичи с учётом накопленной истории.
        """
        df = filtered_df[['source_time', value_col]].copy()
        df = df.sort_values('source_time').reset_index(drop=True)

        # Инициализация истории для нового тега
        if value_col not in self.history:
            self.history[value_col] = deque(maxlen=self.max_history)

        tag_history = self.history[value_col]

        # Создаём расширенный массив: история + новые данные
        all_values = list(tag_history) + df[value_col].tolist()
        extended_series = pd.Series(all_values)

        # Рассчитываем фичи для расширенного ряда
        features_dict = {}

        # 1. Лаговые значения
        for i in range(1, num_lag + 1):
            lag_values = extended_series.shift(i)
            # Берём только последние значения (соответствующие новым данным)
            features_dict[f'{value_col}_lag_{i}'] = lag_values.iloc[-len(df):].fillna(0).tolist()

        # 2. Разности
        diff_values = extended_series.diff(1)
        features_dict[f'{value_col}_diff_1'] = diff_values.iloc[-len(df):].fillna(0).tolist()

        # 3. Скользящие статистики
        for window in [3, 6]:
            # Скользящее среднее
            rolling_mean = extended_series.rolling(window=window, min_periods=1).mean()
            features_dict[f'{value_col}_rolling_mean_{window}'] = rolling_mean.iloc[-len(df):].fillna(0).tolist()

            # Стандартное отклонение
            rolling_std = extended_series.rolling(window=window, min_periods=1).std()
            features_dict[f'{value_col}_rolling_std_{window}'] = rolling_std.iloc[-len(df):].fillna(0).tolist()

            # Дисперсия
            rolling_var = extended_series.rolling(window=window, min_periods=1).var()
            features_dict[f'{value_col}_rolling_var_{window}'] = rolling_var.iloc[-len(df):].fillna(0).tolist()

        # 4. Автокорреляция (упрощённая для реального времени)
        for lag in [3, 6]:
            if len(extended_series) >= lag + 10:  # Минимум данных для расчёта
                try:
                    autocorr_val = extended_series.iloc[-20:].autocorr(lag=lag)
                    if pd.isna(autocorr_val):
                        autocorr_val = 0.0
                except:
                    autocorr_val = 0.0
            else:
                autocorr_val = 0.0

            features_dict[f'{value_col}_autocorr_{lag}'] = [autocorr_val] * len(df)

        # Добавляем фичи в DataFrame
        for feature_name, values in features_dict.items():
            df[feature_name] = values

        # Обновляем историю новыми значениями
        tag_history.extend(df[value_col].tolist())

        return df

# Глобальный экземпляр калькулятора
feature_calculator = FeatureCalculator()

def calculate_features(filtered_df, value_col, num_lag=6):
    """Обёртка для совместимости с существующим кодом."""
    return feature_calculator.calculate_features(filtered_df, value_col, num_lag)
