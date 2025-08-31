# main.py

import asyncio
import logging
from load_data import load_all_tags_into_memory
from forecast import run_forecasting
from config import DATE_BEGIN
from log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def main_pipeline():
    logger.info("🚀 Запуск основного пайплайна")

    stop_event = asyncio.Event()

    loading_task = asyncio.create_task(load_all_tags_into_memory(stop_event))
    forecasting_task = asyncio.create_task(run_forecasting(stop_event))

    try:
        await asyncio.gather(loading_task, forecasting_task)
    except KeyboardInterrupt:
        logger.info("🛑 Программа остановлена пользователем")
        stop_event.set()


if __name__ == '__main__':
    logger.info("✅ Запускаем программу")
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        logger.info("🛑 Остановлено пользователем")

#   venv312\Scripts\activate