# src/data_loader.py

import pandas as pd
import os
from config import (
    logger,
    FORMAT_1_DATA_PATH,
    FORMAT_2_DATA_PATH,
    FORMAT_3_DATA_PATH,
    SELLER_DEFAULT,
)


def load_format1(file_path):
    """
    Загружает данные из старого формата и нормализует под общий вид.
    """
    logger.info(f"📊 [LOAD] Используется формат 1 (старый формат): {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"❌ [LOAD] Не удалось прочитать файл: {e}")
        raise

    column_mapping = {
        'Оценка': 'Оценка',
        'Текст отзыва': 'Текст отзыва',
        'Плюсы': 'Достоинства',
        'Минусы': 'Недостатки',
        'Дата создания': 'Дата создания',
        'Заголовок отзыва': 'Категория товара',
        'Продавец': 'Продавец',
        'Название бренда': 'Название бренда',
        'Название продукта': 'Название товара',
        'Артикул продукта': 'Артикул продавца'
    }

    available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing_columns = set(column_mapping.keys()) - set(df.columns)
    if missing_columns:
        logger.warning(f"⚠️ [LOAD] В исходных данных отсутствуют следующие колонки: {missing_columns}")

    df = df.copy()
    df = df.rename(columns=available_columns)

    new_columns = {
        'ID отзыва': 'нет',
        'Артикул WB': 'нет',
        'Покупатель': 'нет'
    }
    for col, default in new_columns.items():
        if col not in df.columns:
            df[col] = default

    if 'Продавец' in df.columns:
        df['Продавец'] = df['Продавец'].fillna(SELLER_DEFAULT)
    else:
        df['Продавец'] = SELLER_DEFAULT

    if 'Название товара' not in df.columns:
        df['Название товара'] = 'нет'
    if 'Название бренда' not in df.columns:
        df['Название бренда'] = ''
    if 'Артикул продавца' not in df.columns:
        df['Артикул продавца'] = 'нет'

    return df


def load_format2(file_path):
    """
    Загружает данные из нового формата и нормализует под общий вид.
    """
    logger.info(f"📊 [LOAD] Используется формат 2 (новый формат): {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"❌ [LOAD] Не удалось прочитать файл: {e}")
        raise

    needed_columns = {
        'ID отзыва': 'ID отзыва',
        'Дата': 'Дата создания',
        'Артикул продавца': 'Артикул продавца',
        'Артикул WB': 'Артикул WB',
        'Количество звезд': 'Оценка',
        'Бренд': 'Название бренда',
        'Текст отзыва': 'Текст отзыва',
        'Достоинства': 'Достоинства',
        'Недостатки': 'Недостатки',
        'Имя': 'Покупатель'
    }

    present_columns = {k: v for k, v in needed_columns.items() if k in df.columns}
    if not present_columns:
        logger.error("❌ [LOAD] Ни одна из ключевых колонок не найдена в новом формате")
        raise ValueError("Формат файла не соответствует ожиданиям")

    df = df[list(present_columns.keys())].rename(columns=needed_columns)

    optional_columns = {
        'Категория товара': 'нет',
        'Артикул WB': 'нет',
        'Продавец': SELLER_DEFAULT,
        'Название товара': 'нет'
    }
    for col, default in optional_columns.items():
        if col not in df.columns:
            df[col] = default

    if 'Дата создания' in df.columns:
        df['Дата создания'] = df['Дата создания'].astype(str).str.strip()
        df['Дата создания'] = pd.to_datetime(df['Дата создания'], format='%d/%m/%Y', errors='coerce')
        if df['Дата создания'].isna().any():
            df['Дата создания'] = pd.to_datetime(df['Дата создания'], dayfirst=True, errors='coerce')
        df['Дата создания'] = df['Дата создания'].apply(
            lambda x: x.replace(hour=0, minute=1) if pd.notna(x) and isinstance(x, pd.Timestamp) else x
        )

    return df


def load_format3(file_path):
    """
    Загружает данные из третьего формата (ваш специфичный новый формат),
    нормализует под общий вид и удаляет дубликаты по ключевым колонкам,
    также заносит в колонку "Категория товара" данные из "Категория уровня 3".
    """
    logger.info(f"📊 [LOAD] Используется формат 3 (третий формат): {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"❌ [LOAD] Не удалось прочитать файл: {e}")
        raise

    column_mapping = {
        'индекс': 'ID отзыва',
        'покупатель': 'Покупатель',
        'дата отзыва': 'Дата создания',
        'оценка': 'Оценка',
        'достоинства': 'Достоинства',
        'недостатки': 'Недостатки',
        'комментарий': 'Текст отзыва',
        'впечатления покупателя про': 'Впечатления покупателя',
        'артикул': 'Артикул продавца',
        'наименование товара': 'Название товара',
        'категория уровня 1': 'Категория уровня 1',
        'категория уровня 2': 'Категория уровня 2',
        'категория уровня 3': 'Категория уровня 3',
        'бренд': 'Название бренда',
        'название продавца': 'Продавец'
    }

    available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing_columns = set(column_mapping.keys()) - set(df.columns)
    if missing_columns:
        logger.warning(f"⚠️ [LOAD] В исходных данных отсутствуют следующие колонки: {missing_columns}")

    df = df.copy()
    df = df.rename(columns=available_columns)

    defaults = {
        'ID отзыва': 'нет',
        'Покупатель': 'нет',
        'Дата создания': pd.NaT,
        'Оценка': -1,
        'Достоинства': '',
        'Недостатки': '',
        'Текст отзыва': '',
        'Впечатления покупателя': '',
        'Артикул продавца': 'нет',
        'Название товара': 'нет',
        'Категория уровня 1': 'нет',
        'Категория уровня 2': 'нет',
        'Категория уровня 3': 'нет',
        'Название бренда': '',
        'Продавец': SELLER_DEFAULT,
    }
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    # Проставляем в "Категория товара" данные из "Категория уровня 3"
    df['Категория товара'] = df['Категория уровня 3']

    # Обработка даты создания
    df['Дата создания'] = pd.to_datetime(df['Дата создания'], errors='coerce')

    df['Продавец'] = df['Продавец'].fillna(SELLER_DEFAULT)
    df['Покупатель'] = df['Покупатель'].fillna('нет')
    df['Оценка'] = pd.to_numeric(df['Оценка'], errors='coerce').fillna(-1).astype(int)

    # Удаляем дубликаты по ключевым колонкам
    subset = ['Покупатель', 'Дата создания', 'Текст отзыва']
    before_count = len(df)
    df = df.drop_duplicates(subset=subset)
    after_count = len(df)
    logger.info(f"🧹 [LOAD] Удалено дубликатов по колонкам {subset}: {before_count - after_count}")

    return df


def detect_format(file_path):
    """
    Определяет формат файла по наличию ключевых колонок.
    Возвращает:
        'format1' — старый формат
        'format2' — новый формат
        'format3' — третий формат
    """
    try:
        df_sample = pd.read_excel(file_path, nrows=1)
        columns = set(df_sample.columns)

        if {'Оценка', 'Текст отзыва', 'Плюсы', 'Минусы', 'Дата создания'}.issubset(columns):
            return 'format1'
        elif {'ID отзыва', 'Дата', 'Артикул WB', 'Количество звезд', 'Имя'}.issubset(columns):
            return 'format2'
        elif {'индекс', 'покупатель', 'дата отзыва', 'оценка', 'достоинства'}.issubset(columns):
            return 'format3'
        else:
            logger.warning(f"⚠️ [LOAD] Неизвестный формат файла: {file_path}")
            raise ValueError(f"Неизвестный формат файла: {file_path}")
    except Exception as e:
        logger.error(f"❌ [LOAD] Ошибка при определении формата файла — {e}")
        raise


def load_data():
    """
    Универсальная функция загрузки данных.
    Автоматически определяет формат файла и выбирает нужную логику обработки.
    """
    logger.info("🔄 [1] ЗАГРУЗКА ДАННЫХ: начата")

    if os.path.exists(FORMAT_1_DATA_PATH):
        file_to_load = FORMAT_1_DATA_PATH
        data_format = detect_format(file_to_load)

        if data_format == 'format1':
            df = load_format1(file_to_load)
            logger.info(f"✅ [LOAD] Файл загружен с форматом 1, строк: {len(df)}")
            return df
        else:
            logger.warning(f"⚠️ [LOAD] Формат файла {file_to_load} не совпадает с определённым {data_format}")
            raise ValueError(f"Формат файла {file_to_load} не распознан")

    elif os.path.exists(FORMAT_2_DATA_PATH):
        file_to_load = FORMAT_2_DATA_PATH
        data_format = detect_format(file_to_load)

        if data_format == 'format2':
            df = load_format2(file_to_load)
            logger.info(f"✅ [LOAD] Файл загружен с форматом 2, строк: {len(df)}")
            return df
        else:
            logger.warning(f"⚠️ [LOAD] Формат файла {file_to_load} не совпадает с определённым {data_format}")
            raise ValueError(f"Формат файла {file_to_load} не распознан")

    elif os.path.exists(FORMAT_3_DATA_PATH):
        file_to_load = FORMAT_3_DATA_PATH
        data_format = detect_format(file_to_load)

        if data_format == 'format3':
            df = load_format3(file_to_load)
            logger.info(f"✅ [LOAD] Файл загружен с форматом 3, строк: {len(df)}")
            return df
        else:
            logger.warning(f"⚠️ [LOAD] Формат файла {file_to_load} не совпадает с определённым {data_format}")
            raise ValueError(f"Формат файла {file_to_load} не распознан")

    else:
        logger.error(f"❌ [1] ЗАГРУЗКА ДАННЫХ: ни один файл не найден в директории data/")
        raise FileNotFoundError("Ни один файл не найден в директории data/")
