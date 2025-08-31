# src/decorators.py

import logging
from functools import wraps

# Создаём локальный логгер
logger = logging.getLogger("pipeline.decorators")


def pipeline_step(step_number, step_name, log_start=True, log_result=True):
    """
    Декоратор для унификации шагов пайплайна

    :param step_number: номер шага
    :param step_name: название шага
    :param log_start: логировать начало
    :param log_result: логировать результат
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, stop_pipeline_flag=False, **kwargs):
            if stop_pipeline_flag:
                logger.warning(f"🔚 [INTERRUPT] Пропуск шага [{step_number}] {step_name} — остановлен пользователем")
                return None

            try:
                result = func(*args, stop_pipeline_flag=stop_pipeline_flag, **kwargs)
                return result

            except Exception as e:
                logger.error(f"❌ [{step_number}] {step_name}: выполнение прервано из-за ошибки — {e}")
                return None

        return wrapper

    return decorator