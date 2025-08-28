'''
pip install psycopg2 pyarrow pandas
pip install psycopg2-binary pyarrow pandas openpyxl

pip install tensorflow
pip install scikit-learn
pip install joblib

'''

# config.py

from datetime import datetime, timedelta, timezone
import os


# === Режим тестирования загрузки данных ===
TEST_LOAD_DATA = True  # Сохраняем сырые данные после загрузки из БД
TEST_DATA_SEC = True   # Сохраняем данные после группировки по секундам и восстановления пропущенных секунд
TEST_DATA_GROUP = True    # Сохраняем после объединения всех тегов и замены отрицательных значений "0"
TEST_DATA_DISCRET = True   # Сохраняем после дискретизации по CHUNK_SECONDS
TEST_DATA_CALC = True   # Сохраняем расчётные фичи по тегам

# === Пути проекта ===
WORK_DIR = r"C:\Users\Андрей\Documents\Курсы\AI_ML_разработчик\Стажировка_1\Задача_1\Интеграция_2"
RESULT_DIR = os.path.join(WORK_DIR, "result")
DATA_DIR = os.path.join(WORK_DIR, "data")
MODEL_DIR = os.path.join(WORK_DIR, "models")
LOGS_DIR = os.path.join(WORK_DIR, "logs")

for dir_path in [WORK_DIR, RESULT_DIR, DATA_DIR, MODEL_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# === Пользовательские настройки ===
TAGS = [10, 14, 16]  # теги для прогнозирования
CHUNK_SECONDS = 10     # шаг между точками
LOOK_BACK = 10         # размер окна модели
FORECAST_HORIZON = 30   # горизонт прогноза (шагов)
MIN_REQUIRED = LOOK_BACK + FORECAST_HORIZON  # минимальное количество записей

# === Часовой пояс и временные параметры ===
TIMEZONE_OFFSET = 3  # UTC+3 (Москва)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# === Временной диапазон: только DATE_BEGIN ===
def get_initial_time_range():
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=30)
    return start_time.strftime(DATE_FORMAT), end_time.strftime(DATE_FORMAT)

DATE_BEGIN, DATE_NOW = get_initial_time_range()  # ❌ DATE_NOW уберём из логики работы

# === Интервал между итерациями ===
SLEEP_INTERVAL = 1  # секунд


# === Настройки подключения к БД ===
DB_HOST = "emcable-and-amai.cyberb2b.ru"  # ✅ Добавлено
DB_PORT = 15432                           # ✅ Добавлено
DB_NAME = "mscada_db"                     # ✅ Добавлено
DB_USER = "student"                        # ✅ Добавлено
DB_PASSWORD = "5SxdeChZ"                   # ✅ Добавлено

# === Путь к модели и скалерам ===
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
FEATURES_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_features.pkl')
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_target.pkl')

# === Список 48 фичей, на которых обучалась модель ===
REQUIRED_FEATURES = [
    'value_14', 'value_10', 'value_16',

    'value_10_lag_1', 'value_10_lag_2', 'value_10_lag_3', 'value_10_lag_4', 'value_10_lag_5', 'value_10_lag_6',
    'value_10_diff_1', 'value_10_rolling_mean_3', 'value_10_rolling_mean_6',
    'value_10_autocorr_3', 'value_10_autocorr_6', 'value_10_rolling_var_3', 'value_10_rolling_var_6',
    'value_10_rolling_std_3', 'value_10_rolling_std_6',

    'value_14_lag_1', 'value_14_lag_2', 'value_14_lag_3', 'value_14_lag_4', 'value_14_lag_5', 'value_14_lag_6',
    'value_14_diff_1', 'value_14_rolling_mean_3', 'value_14_rolling_mean_6',
    'value_14_autocorr_3', 'value_14_autocorr_6', 'value_14_rolling_var_3', 'value_14_rolling_var_6',
    'value_14_rolling_std_3', 'value_14_rolling_std_6',

    'value_16_lag_1', 'value_16_lag_2', 'value_16_lag_3', 'value_16_lag_4', 'value_16_lag_5', 'value_16_lag_6',
    'value_16_diff_1', 'value_16_rolling_mean_3', 'value_16_rolling_mean_6',
    'value_16_autocorr_3', 'value_16_autocorr_6', 'value_16_rolling_var_3', 'value_16_rolling_var_6',
    'value_16_rolling_std_3', 'value_16_rolling_std_6'
]