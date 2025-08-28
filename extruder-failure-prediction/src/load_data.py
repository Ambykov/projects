# load_data.py

import asyncio
import pandas as pd
from functools import reduce
from export_data_by_tag import export_data_by_tag
from clean_data import clean_dataframe
from calculate_features import calculate_features
from config import (
    TAGS,
    RESULT_DIR,
    TEST_LOAD_DATA,
    TEST_DATA_SEC,
    TEST_DATA_GROUP,
    TEST_DATA_DISCRET,
    TEST_DATA_CALC,
    SLEEP_INTERVAL,
    TIMEZONE_OFFSET,
    CHUNK_SECONDS,
    DATE_FORMAT,
    REQUIRED_FEATURES
)
import logging
import os
from datetime import datetime, timedelta, timezone
import numpy as np

logger = logging.getLogger(__name__)


async def fetch_tag_data(tag):
    """Загружает данные для одного тега и дописывает только новые значения"""
    try:
        df = export_data_by_tag(tag)  # ❗ Синхронный вызов
        if df is not None and not df.empty:
            logger.info(f"[INFO] 📥 Получены данные для тега {tag}: {df.shape[0]} строк")

            # Преобразуем source_time к datetime
            df['source_time'] = pd.to_datetime(df['source_time'])

            # Путь к файлу
            raw_file = os.path.join(RESULT_DIR, f"data_load_tag_{tag}.csv")

            # Читаем только последнюю метку времени из существующего файла (если есть)
            last_saved_time = None
            if os.path.exists(raw_file):
                # Чтение только нужных столбцов и последней строки
                try:
                    last_row = pd.read_csv(raw_file, parse_dates=['source_time'],
                                           usecols=['source_time'],
                                           tail=1)
                    if not last_row.empty:
                        last_saved_time = last_row.iloc[-1]['source_time']
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка чтения последней строки: {e}")

            # Фильтруем новые данные: оставляем только те, что позже последней сохранённой
            if last_saved_time is not None:
                new_data = df[df['source_time'] > last_saved_time]
            else:
                new_data = df  # если файл новый или пустой — сохраняем всё

            # Если новых данных нет — выходим
            if new_data.empty:
                logger.debug(f"[DEBUG] Нет новых данных для тега {tag}")
                return df[['source_time', f'value_{tag}']].copy()

            # Дописываем только новые данные
            file_exists = os.path.exists(raw_file)
            new_data.to_csv(raw_file, index=False, mode='a', header=not file_exists)
            logger.debug(f"[DEBUG] ✅ Добавлено новых записей для тега {tag}: {new_data.shape[0]}")

            return df[['source_time', f'value_{tag}']].copy()
        else:
            logger.warning(f"⚠️ Нет данных для тега {tag}")
            return None
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных для тега {tag}: {e}")
        return None



def get_new_data_only(filepath, new_data):
    """
    Возвращает только те строки из new_data,
    у которых source_time > последней метки в файле.
    """
    last_saved_time = None
    if os.path.exists(filepath):
        try:
            # Читаем только source_time из последней строки
            last_row = pd.read_csv(
                filepath,
                parse_dates=['source_time'],
                usecols=['source_time'],
                tail=1
            )
            if not last_row.empty:
                last_saved_time = pd.to_datetime(last_row.iloc[-1]['source_time'])
                logger.debug(f"[DEBUG] 🕒 Последнее время в {filepath}: {last_saved_time}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения последней строки из {filepath}: {e}")

    # Приводим source_time в new_data к datetime
    new_data['source_time'] = pd.to_datetime(new_data['source_time'])

    if last_saved_time is not None:
        filtered = new_data[new_data['source_time'] > last_saved_time]
        logger.debug(f"[DEBUG] ✨ Новых записей для {filepath}: {len(filtered)}")
        return filtered
    else:
        logger.debug(f"[DEBUG] 🆕 Создаём новый файл: {filepath}")
        return new_data


async def process_tag_data(df, tag):
    """Обрабатывает данные по одному тегу:
    - Чистка по секундам
    - Восстановление временного ряда
    - Расчёт фичей
    """
    if df is None or df.empty:
        logger.warning(f"⚠️ Нет данных для обработки тега {tag}")
        return None
    logger.info(f"⚙️ Обрабатываем тег {tag}")
    # Шаг 1: чистка датафрейма по секундам
    cleaned_df = clean_dataframe(df, f'value_{tag}')
    logger.debug(f"[DEBUG] 📉 После очистки тега {tag}: {cleaned_df.head()}")
    logger.info(f"[INFO] 📏 Размер cleaned_df для тега {tag}: {cleaned_df.shape}")

    # Шаг 2: восстанавливаем полный временной ряд с шагом 1s
    full_range = pd.date_range(
        start=cleaned_df['source_time'].min(),
        end=cleaned_df['source_time'].max(),
        freq='s'
    )
    reindexed_df = cleaned_df.set_index('source_time').reindex(full_range).ffill().reset_index()
    reindexed_df.rename(columns={'index': 'source_time'}, inplace=True)
    logger.debug(f"[DEBUG] ⏱️ Временной ряд восстановлен для тега {tag}: {reindexed_df.head()}")

    # Шаг 3: замена отрицательных значений на "0"
    reindexed_df[f'value_{tag}'] = reindexed_df[f'value_{tag}'].clip(lower=0)
    logger.debug(f"[DEBUG] 📉 Отрицательные значения тега {tag} заменены на 0: {reindexed_df[[f'value_{tag}']].head()}")


    # Шаг 4: сохранение после группировки по секундам (тестирование)
    if TEST_DATA_SEC:
        sec_file = os.path.join(RESULT_DIR, f"data_sec_tag_{tag}.csv")
        debug_df = reindexed_df.copy()

        # Получаем только новые данные
        new_sec_data = get_new_data_only(sec_file, debug_df[['source_time', f'value_{tag}']].copy())

        if not new_sec_data.empty:
            file_exists = os.path.exists(sec_file)
            new_sec_data.to_csv(sec_file, index=False, mode='a', header=not file_exists)
            logger.debug(f"[DEBUG] ✅ Записано {len(new_sec_data)} новых строк в {sec_file}")
        else:
            logger.debug(f"[DEBUG] ⚠️ Нет новых данных для записи в {sec_file}")


    # Шаг 4: дискретизация по CHUNK_SECONDS
    reindexed_df['source_time'] = pd.to_datetime(reindexed_df['source_time'])
    discretized_df = reindexed_df.resample(f'{CHUNK_SECONDS}s', on='source_time').last().reset_index()

    numeric_cols = discretized_df.select_dtypes(include=np.number).columns
    discretized_df[numeric_cols] = discretized_df[numeric_cols].ffill().bfill()

    # Сохранение дискретизированных данных (опционально)
    if TEST_DATA_DISCRET:
        discret_file = os.path.join(RESULT_DIR, f"data_discrete_tag_{tag}.csv")
        debug_df = discretized_df[['source_time', f'value_{tag}']].copy()
        debug_df['source_time'] = debug_df['source_time'].dt.strftime(DATE_FORMAT)
        file_exists = os.path.exists(discret_file)
        debug_df.to_csv(discret_file, index=False, mode='a', header=not file_exists)
        logger.debug(f"[DEBUG] 📁 Данные после дискретизации записаны: {discret_file}")

    return discretized_df[['source_time', f'value_{tag}']]


async def load_all_tags_into_memory(stop_event):
    logger.info("🚀 Начинаем потоковую загрузку данных в реальном времени")
    while not stop_event.is_set():
        tasks = [fetch_tag_data(tag) for tag in TAGS]
        dfs = await asyncio.gather(*tasks)

        all_features_dfs = []

        for df, tag in zip(dfs, TAGS):
            if df is None or df.empty:
                logger.warning(f"⚠️ Нет данных для тега {tag}, пропускаем")
                continue

            # Обработка данных по одному тегу
            discretized_df = await process_tag_data(df, tag)
            if discretized_df is None or discretized_df.empty:
                continue

            # Расчёт фичей
            value_col = f'value_{tag}'
            features_df = calculate_features(discretized_df, value_col)
            if features_df.empty:
                logger.warning(f"⚠️ Не удалось рассчитать фичи для тега {tag}")
                continue

            # Сохранение фичей (опционально)
            if TEST_DATA_CALC:
                calc_file = os.path.join(RESULT_DIR, f"data_calculated_tag_{tag}.csv")
                debug_df = features_df.copy()
                debug_df['source_time'] = pd.to_datetime(debug_df['source_time']).dt.strftime(DATE_FORMAT)
                file_exists = os.path.exists(calc_file)
                debug_df.to_csv(calc_file, index=False, mode='a', header=not file_exists)
                logger.debug(f"[DEBUG] 📁 Фичи для тега {tag} записаны: {calc_file}")

            all_features_dfs.append(features_df)

        if not all_features_dfs:
            logger.warning("⚠️ Нет фичей для прогнозирования")
            await asyncio.sleep(SLEEP_INTERVAL)
            continue

        # Объединяем все фичи по source_time
        final_df = reduce(lambda left, right: pd.merge(left, right, on='source_time', how='outer'), all_features_dfs)
        final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()
        final_df.sort_values('source_time', inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        # Проверка наличия всех необходимых фичей
        missing_features = set(REQUIRED_FEATURES) - set(final_df.columns)
        if missing_features:
            logger.warning(f"[WARN] ❌ Отсутствуют фичи: {missing_features}. Прогноз невозможен.")
            logger.info("🔄 Ждём новых данных...")
            await asyncio.sleep(SLEEP_INTERVAL)
            continue

        # Сохранение в файл calculated_features.csv
        output_path = os.path.join(RESULT_DIR, "calculated_features.csv")
        final_df_with_time = final_df[['source_time'] + REQUIRED_FEATURES].copy()
        final_df_with_time['source_time'] = pd.to_datetime(final_df_with_time['source_time']).dt.strftime(DATE_FORMAT)
        final_df_with_time['source_time'] = pd.to_datetime(final_df_with_time['source_time'])

        if os.path.exists(output_path):
            prev_df = pd.read_csv(output_path, parse_dates=['source_time'])
            combined_df = pd.concat([prev_df, final_df_with_time], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['source_time'], keep='last')
        else:
            combined_df = final_df_with_time.copy()

        combined_df.sort_values('source_time', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        new_records_count = final_df_with_time.shape[0]
        total_records_count = combined_df.shape[0]

        try:
            combined_df.to_csv(output_path, index=False)
            logger.info(f"✅ Признаки обновлены: {output_path}")
            logger.info(f"🆕 Добавлено записей: {new_records_count} | 📦 Всего записей в файле: {total_records_count}")
        except Exception as e:
            logger.error(f"❌ Ошибка при записи в calculated_features.csv: {e}")

        await asyncio.sleep(SLEEP_INTERVAL)
    logger.info("🏁 Загрузка данных завершена.")