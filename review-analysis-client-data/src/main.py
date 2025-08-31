# src/main.py

import sys
import os
import signal
import logging

# --- Настройка пути к проекту ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Логгер ---
logger = logging.getLogger('pipeline')

# --- Импорты классификаторов ---
from preprocess import preprocess_reviews_with_loading
from classifiers.classifier_0 import classifier_0
from classifiers.classifier_100 import classifier_100
from classifiers.classifier_200 import classifier_200
from classifiers.classifier_201 import classifier_201
from classifiers.classifier_202 import classifier_202
from classifiers.classifier_300 import classifier_300
from classifiers.classifier_500 import classifier_500
from classifiers.classifier_700 import classifier_700
from classifiers.classifier_701 import classifier_701
from classifiers.classifier_result import classifier_result



#--- Импорт фильтра отзывов по заданному уровню оценки ---
from utils import filter_by_negativ_level
from utils import step_clean_text

# --- Флаг прерывания ---
stop_pipeline = False


class PipelineInterruptError(Exception):
    pass


def signal_handler(sig, frame):
    global stop_pipeline
    logger.warning("🛑 [INTERRUPT] Получен сигнал прерывания от пользователя (Ctrl+C)")
    stop_pipeline = True
    raise PipelineInterruptError()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Список шагов ---
PIPELINE_STEPS = [
    {
        "name": "ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА",
        "function": preprocess_reviews_with_loading,
        "step_number": 1,
        "enabled": True,
        "log_message": "⚙️  [1] ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА: старт - из main"
    },
    {
        "name": "НАЧАЛЬНАЯ КЛАССИФИКАЦИЯ",
        "function": classifier_0,
        "step_number": 2,
        "enabled": True,
        "log_message": "🏷️  [2] НАЧАЛЬНАЯ КЛАССИФИКАЦИЯ: старт"
    },
    {
        "name": "ПРОВЕРКА НА [100]",
        "function": classifier_100,
        "step_number": 3,
        "enabled": True,
        "save_flag": "SAVE_STEP_3_RESULT",
        "log_message": "🔍  [3] ПРОВЕРКА НА [100]: старт"
    },
     {
        "name": "КЛАССИФИКАТОР [200] — Поиск ссылок на сайты и соцсети",
        "function": classifier_200,
        "step_number": 4,
        "enabled": True,
        "save_flag": "SAVE_STEP_4_RESULT",
        "log_message": "🔍  [4] КЛАССИФИКАТОР [200]: старт"
    },
    {
        "name": "ПОИСК ДУБЛИКАТОВ → [300] СПАМ",
        "function": classifier_300,
        "step_number": 5,
        "enabled": True,
        "save_flag": "SAVE_STEP_5_RESULT",
        "log_message": "🗂️  [5] ПОИСК ДУБЛИКАТОВ: старт"
    },
    {
        "name": "ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ → [500] НЕЦЕНЦУРНАЯ ЛЕКСИКА",
        "function": classifier_500,
        "step_number": 6,
        "enabled": True,
        "save_flag": "SAVE_STEP_6_RESULT",
        "log_message": "🔑  [6] ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ [500]: старт"
    },
    {
        "name": "ОЧИСТКА ТЕКСТА",
        "function": step_clean_text,
        "step_number": 7,
        "enabled": True,
        "save_flag": "SAVE_STEP_7_RESULT",
        "log_message": "🧹  [7] ОЧИСТКА ТЕКСТА: старт"
    },
    {
        "name": "ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ [201] - Б/У",
        "function": classifier_201,
        "step_number": 8,
        "enabled": True,
        "save_flag": "SAVE_STEP_8_RESULT",
        "log_message": "🔍  [8] КЛАССИФИКАТОР [201]: старт"
    },
    {
        "name": "ПОИСК ПОЛОЖИТЕЛЬНЫХ ОТЗЫВОВ С НЕГАТИВНОЙ ОЦЕНКОЙ→ [202]",
        "function": classifier_202,
        "step_number": 9,
        "enabled": True,
        "save_flag": "SAVE_STEP_9_RESULT",
        "log_message": "🔍  [9] КЛАССИФИКАТОР [202]: старт"
    },
    {
        "name": "ПОИСК ПОЛИТ. КОНТЕНТА ПО КЛЮЧЕВЫМ ФРАЗАМ [700]",
        "function": classifier_700,
        "step_number": 10,
        "enabled": True,
        "save_flag": "SAVE_STEP_10_RESULT",
        "log_message": "🧠  [10] КЛАССИФИКАТОР [700]: старт"
    },
    {
        "name": "ПОИСК ПОЛИТ. КОНТЕНТА МОДЕЛЬЮ [701]",
        "function": classifier_701,
        "step_number": 11,
        "enabled": True,
        "save_flag": "SAVE_STEP_11_RESULT",
        "log_message": "🧠  [11] КЛАССИФИКАТОР [701]: старт"
    },
    {
        "name": "ИТОГОВАЯ КЛАССИФИКАЦИЯ — КЛАССИФИКАТОР RESULT",
        "function": classifier_result,
        "step_number": 12,
        "enabled": True,
        "save_flag": "SAVE_STEP_12_RESULT",
        "log_message": "🏷️  [12] Итоговый классификатор RESULT: старт"
    },
]


def run_pipeline():
    logger.info("🔵 🔁 === Начало выполнения пайплайна === 🔁 🔵")

    result_df = None

    for step in PIPELINE_STEPS:
        if not step["enabled"]:
            logger.warning(f"🟡 [!] Шаг [{step['step_number']}] {step['name']} отключён")
            continue

        if stop_pipeline:
            logger.critical("⛔ [!] Пайплайн остановлен пользователем")
            break

        logger.info(step["log_message"])

        try:
            result_df = step["function"](result_df, stop_pipeline_flag=stop_pipeline)

            if result_df is None:
                logger.critical(f"⛔ [!] Шаг [{step['step_number']}] {step['name']} вернул None — выполнение остановлено")
                break

            logger.info(f"✅ [{step['step_number']}] {step['name']}: успешно выполнен")

        except PipelineInterruptError:
            logger.critical("⛔ [!] Пайплайн остановлен пользователем")
            break

        except Exception as e:
            logger.error(f"❌ [{step['step_number']}] Ошибка на шаге {step['name']} — {e}")
            logger.critical(f"⛔ [!] Шаг [{step['step_number']}] {step['name']} прерван. Выполнение остановлено.")
            break

    logger.info("🟢 🎉 === Пайплайн успешно завершён === 🎉 🟢")


if __name__ == "__main__":
    run_pipeline()