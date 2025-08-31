# log_config.py

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from config import LOGS_DIR

def setup_logging(log_filename=None):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    level = logging.DEBUG
    format_str = '%(asctime)s [%(levelname)-5.5s] %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_str))
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_filename is None:
        log_filename = os.path.join(LOGS_DIR, "pipeline.log")

    file_handler = RotatingFileHandler(log_filename, maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(module)s: %(message)s'))
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    return root_logger