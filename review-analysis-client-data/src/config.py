import os
import sys
import logging
import colorlog
from datetime import datetime

# --- Базовые пути ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def path(*relative_parts):
    """
    Формирует абсолютный путь, объединяя PROJECT_ROOT и относительные части.
    """
    return os.path.join(PROJECT_ROOT, *relative_parts)

# --- Директории проекта ---
DATA_DIR = path("..", "data")
RESULTS_DIR = path("..", "results")
KEY_WORDS_DIR = path("..", "key_words")
MODELS_DIR = path("..", "models")
LOGS_DIR = path("..", "logs")
PROCESSED_DIR = path("..", "processed")

# Создаем папки, если они ещё не существуют
for dir_path in [DATA_DIR, RESULTS_DIR, KEY_WORDS_DIR, MODELS_DIR, LOGS_DIR, PROCESSED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Файлы данных ---
FORMAT_1_DATA_PATH = os.path.join(DATA_DIR, "format1_reviews.xlsx")  # Формат 1: старый формат
FORMAT_2_DATA_PATH = os.path.join(DATA_DIR, "format2_reviews.xlsx")  # Формат 2: новый формат
FORMAT_3_DATA_PATH = os.path.join(DATA_DIR, "format3_reviews.xlsx")  # Формат 3: распарсенные данные с сайта вайлдберриз
PROCESSED_DATA_PATH = os.path.join(RESULTS_DIR, "processed_reviews.xlsx")

# --- Пути к моделям ---
MODEL_701_DIR = os.path.join(MODELS_DIR, "cnn_bilstm_attention_27072025_1216")
WORD2IDX_701_PATH = os.path.join(MODEL_701_DIR, "word2idx.pkl")
EMBED_MATRIX_701_PATH = os.path.join(MODEL_701_DIR, "embed_matrix.npy")
MODEL_701_WEIGHTS_PATH = os.path.join(MODEL_701_DIR, "model_20250727_1216.pt")

# --- Глобальные параметры ---
SELLER_DEFAULT = "giper_fm"  # Значение по умолчанию для поля 'Продавец'

# --- Уровень негативности ---
NEGATIV_LEVEL = 2  # Например, фильтруем отзывы с оценкой 1 и 2

# --- Код класса для отзывов с оценкой выше NEGATIV_LEVEL ---
FILTERED_OUT_CLASS = 999

# --- Коды классов ---
CLASS_100 = 100

# --- Ключевые слова ---
KEYWORDS_FILES = {
    500: os.path.join(KEY_WORDS_DIR, "key_words_500.py"),
    700: os.path.join(KEY_WORDS_DIR, "key_words_700.py"),
    201: os.path.join(KEY_WORDS_DIR, "key_words_201.py"),
}

# --- Временная метка для файлов ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Пути для сохранения результатов последнего шага (12) ---
STEP_12_RESULT_FILE = os.path.join(PROCESSED_DIR, f"step_12_result_classification_{TIMESTAMP}.xlsx")

# --- Настройки логирования ---
log_filename = f"pipeline_{TIMESTAMP}.log"
log_filepath = os.path.join(LOGS_DIR, log_filename)

logger = colorlog.getLogger('pipeline')
logger.setLevel(logging.DEBUG)  # Уровень логирования для логгера

# Цветной вывод в консоль
console_handler = colorlog.StreamHandler()
console_formatter = colorlog.ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    },
    style='%'
)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)  # В консоль выводим info+
logger.addHandler(console_handler)

# Логирование в файл без цвета
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", style='%')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
logger.addHandler(file_handler)

# --- Флаги сохранения результатов по шагам ---
SAVE_STEP_1_RESULT = True    # Шаг [1]
SAVE_STEP_3_RESULT = True    # После классификатора 100
SAVE_STEP_4_RESULT = True    # После классификатора 200
SAVE_STEP_5_RESULT = True    # После классификатора 300
SAVE_STEP_6_RESULT = True    # После классификатора 500
SAVE_STEP_8_RESULT = True    # После классификатора 201
SAVE_STEP_9_RESULT = True    # После классификатора 202
SAVE_STEP_10_RESULT = True   # После классификатора 700
SAVE_STEP_11_RESULT = True   # После классификатора 701
SAVE_STEP_12_RESULT = True   # После итоговой классификации
