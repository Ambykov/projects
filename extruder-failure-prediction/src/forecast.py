# forecast.py

import asyncio
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from datetime import datetime, timedelta

from config import (
    MODEL_PATH,
    FEATURES_SCALER_PATH,
    TARGET_SCALER_PATH,
    LOOK_BACK,
    FORECAST_HORIZON,
    REQUIRED_FEATURES,
    RESULT_DIR,
    SLEEP_INTERVAL
)

logger = logging.getLogger(__name__)

# === Загрузка модели и скалеров === #
try:
    model = load_model(MODEL_PATH)
    logger.info("✅ Модель загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    raise

try:
    with open(FEATURES_SCALER_PATH, 'rb') as f:
        scaler_features = joblib.load(f)
    logger.info("✅ Скалер признаков загружен")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки скалера признаков: {e}")
    raise

try:
    with open(TARGET_SCALER_PATH, 'rb') as f:
        scaler_target = joblib.load(f)
    logger.info("✅ Скалер целевой переменной загружен")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки скалера целевой переменной: {e}")
    raise


def calculate_deviation(real, predicted):
    """Вычисляет абсолютное и относительное отклонение"""
    if real is None or np.isnan(real) or np.isnan(predicted):
        return np.nan, np.nan
    abs_dev = abs(real - predicted)
    rel_dev = abs_dev / real if real != 0 else 0
    return abs_dev, rel_dev


async def predict_forecasts():
    """
    Прогнозирует значение value_14 на основе последних данных
    Сохраняет последние 30 прогнозов в forecasts.csv
    """

    input_path = os.path.join(RESULT_DIR, "calculated_features.csv")
    output_path = os.path.join(RESULT_DIR, "forecasts.csv")

    # === Проверяем существование файла calculated_features.csv === #
    if not os.path.exists(input_path):
        logger.warning("⚠️ Файл calculated_features.csv не найден. Ждём новых данных...")
        return

    try:
        input_df = pd.read_csv(input_path)
        input_df['source_time'] = pd.to_datetime(input_df['source_time'])
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла: {e}")
        return

    # Убираем дубликаты колонок
    input_df = input_df.loc[:, ~input_df.columns.duplicated()]
    input_df.sort_values('source_time', inplace=True)
    input_df.reset_index(drop=True, inplace=True)

    # === Проверяем наличие всех фичей === #
    missing_features = set(REQUIRED_FEATURES) - set(input_df.columns)
    if missing_features:
        logger.warning(f"[WARN] Отсутствуют фичи: {missing_features}. Прогнозирование невозможно.")
        return

    # === Подготавливаем матрицу признаков === #
    X_data = input_df[REQUIRED_FEATURES].values
    y_data = input_df['value_14'].values

    if len(X_data) < LOOK_BACK + FORECAST_HORIZON:
        logger.warning(f"⏳ Недостаточно данных для формирования окна ({len(X_data)} / {LOOK_BACK + FORECAST_HORIZON})")
        return

    # === Формируем список окон === #
    windows = []
    prediction_times = []

    for i in range(LOOK_BACK, len(X_data) - FORECAST_HORIZON + 1):
        window = X_data[i - LOOK_BACK:i]
        prediction_times.append(input_df.iloc[i]['source_time'])
        windows.append(window)

    if not windows:
        logger.warning("⚠️ Нет окон для прогноза")
        return

    # === Масштабируем данные === #
    try:
        scaled_X = scaler_features.transform(np.concatenate(windows))
        reshaped_scaled_X = scaled_X.reshape(len(windows), LOOK_BACK, -1)
    except ValueError as e:
        logger.error(f"❌ Ошибка масштабирования: {e}")
        return

    # === Прогнозируем значения === #
    predictions = model.predict(reshaped_scaled_X, verbose=0)
    predicted_values = scaler_target.inverse_transform(predictions).flatten()

    # === Время прогноза (через FORECAST_HORIZON шагов × 10 секунд) === #
    forecast_times = [t + timedelta(seconds=FORECAST_HORIZON * 10) for t in prediction_times]

    # === Получаем фактические значения через FORECAST_HORIZON шагов назад === #
    actual_indices = [i + FORECAST_HORIZON for i in range(len(prediction_times))]
    actual_values = [y_data[idx] if idx < len(y_data) else np.nan for idx in actual_indices]

    # === Рассчитываем отклонения === #
    deviations = [calculate_deviation(actual, pred) for actual, pred in zip(actual_values, predicted_values)]

    # === Берём только последние 30 прогнозов === #
    min_len = min(
        len(prediction_times),
        len(forecast_times),
        len(predicted_values),
        len(actual_values),
        len([d[0] for d in deviations]),
        len([d[1] for d in deviations])
    )

    forecasts_df = pd.DataFrame({
        'prediction_time': prediction_times[-min_len:],
        'forecast_time': forecast_times[-min_len:],
        'predicted_value_14': predicted_values[-min_len:],
        'actual_value_14': actual_values[-min_len:]
    })

    # === Добавляем временную зону к меткам времени === #
    #forecasts_df['prediction_time'] += timedelta(hours=3)
    #forecasts_df['forecast_time'] += timedelta(hours=3)

    # === Рассчитываем отклонения после обрезки до min_len === #
    abs_devs = [d[0] for d in deviations][-min_len:]
    rel_devs = [d[1] for d in deviations][-min_len:]

    # === Обновляем датафрейм с результатами === #
    forecasts_df['abs_deviation'] = abs_devs
    forecasts_df['rel_deviation'] = rel_devs

    # === Сохраняем только последние 30 строк === #
    final_forecasts = forecasts_df.tail(30)

    # === Сохраняем в файл === #
    file_exists = os.path.exists(output_path)
    final_forecasts.to_csv(output_path, index=False, mode='a', header=not file_exists)
    logger.info(f"✅ Прогноз сохранён: {output_path}")


async def run_forecasting(stop_event):
    logger.info("🚀 Прогнозирование запущено")
    while not stop_event.is_set():
        await predict_forecasts()
        await asyncio.sleep(SLEEP_INTERVAL)