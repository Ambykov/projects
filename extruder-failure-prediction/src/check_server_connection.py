# check_server_connection.py
import socket
import logging

logger = logging.getLogger(__name__)

def check_server_connection(host, port, timeout=5):
    """
    Проверяет доступность сервера и порта.
    :param host: Хост БД
    :param port: Порт БД
    :param timeout: Таймаут ожидания подключения (в секундах)
    :return: True, если сервер доступен, иначе False
    """
    try:
        logger.info(f"[INFO] Проверяю подключение к {host}:{port}")
        with socket.create_connection((host, port), timeout=timeout):
            logger.info(f"[SUCCESS] ✅ Сервер {host}:{port} доступен")
            return True
    except OSError as e:
        logger.error(f"[ERROR] ❌ Невозможно подключиться к {host}:{port} → {e}", exc_info=True)
        return False
