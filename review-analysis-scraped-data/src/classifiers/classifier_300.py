# src/classifiers/classifier_300.py

import logging
import os
import pandas as pd
from tqdm import tqdm

from utils import set_class, class_statistics
from decorators import pipeline_step
from config import logger, SAVE_STEP_5_RESULT, PROCESSED_DIR, TIMESTAMP, FILTERED_OUT_CLASS, CLASS_100


@pipeline_step(step_number=5, step_name="ПОИСК ДУБЛИКАТОВ → [300]")
def classifier_300(df, stop_pipeline_flag=False, step_number=5):
    """
    Присваивает класс [300], если отзыв является дубликатом внутри дня.
    Причина пометки: дубль к строке + время оригинала
    """

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной датафрейм пустой или None")
        return df

    # Убедимся, что колонка 'Примечание' существует
    if 'Примечание' not in df.columns:
        df['Примечание'] = ''

    # Определение функции для игнорируемых классов
    def has_ignored_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes or CLASS_100 in classes
        return classes == FILTERED_OUT_CLASS or classes == CLASS_100

    # --- Подготовка данных для группировки ---
    # Временные колонки добавляются прямо в основной df, т.к. по нему и будем группировать
    df['Дата создания'] = pd.to_datetime(df['Дата создания'], errors='coerce')
    df['date_only'] = df['Дата создания'].dt.date
    df['group_key'] = df['Отзыв'].astype(str) + ' | ' + df['Категория товара'].astype(str)

    group_cols = ['date_only', 'group_key', 'Продавец']
    duplicates_count = 0
    detected_duplicates = []

    # Итерируемся по группам основного DataFrame
    for group_keys, group in tqdm(df.groupby(group_cols), total=df[group_cols].nunique(dropna=False).prod(), desc="🔍 Поиск дубликатов"):
        if stop_pipeline_flag:
            logger.warning(f"🔚 [{step_number}] Шаг прерван внутри цикла")
            break

        # Фильтруем текущую группу, исключая отзывы с уже проставленными игнорируемыми классами
        # (это важно, т.к. df['Класс'] может меняться в предыдущих классификаторах или на этом шаге)
        filtered_group = group[~group['Класс'].apply(has_ignored_class)]

        if len(filtered_group) > 1:
            # Отсортируем отфильтрованную группу по дате создания
            sorted_filtered_group = filtered_group.sort_values(by='Дата создания')
            indices_sorted = sorted_filtered_group.index.tolist()

            first_idx = indices_sorted[0]
            duplicate_indices = indices_sorted[1:]  # Все индексы, кроме первого (оригинала)

            for idx in duplicate_indices:
                # Дополнительная, но уже избыточная проверка, если has_ignored_class применяется к filtered_group
                # current_class = df.at[idx, 'Класс']
                # if isinstance(current_class, list) and (FILTERED_OUT_CLASS in current_class or CLASS_100 in current_class):
                #     continue

                df = set_class(df, idx, code=300)
                original_time = str(df.at[first_idx, 'Дата создания'])

                old_note = df.at[idx, 'Примечание']
                new_duplicate_note = f"дубль к строке {first_idx}, время оригинала {original_time}"
                df.at[idx, 'Примечание'] = f"{old_note} | {new_duplicate_note}" if old_note else new_duplicate_note

                duplicates_count += 1

                detected_duplicates.append({
                    'Номер строки': idx,
                    'Отзыв': df.at[idx, 'Отзыв'],
                    'Дата создания': str(df.at[idx, 'Дата создания']),
                    'Продавец': df.at[idx, 'Продавец'],
                    'Дубль к строке': first_idx,
                    'Время оригинала': original_time
                })

    # --- Удаляем временные колонки ---
    df.drop(columns=['date_only', 'group_key'], inplace=True, errors='ignore')

    # --- Логируем результаты ---
    logger.info(f"✅ [{step_number}] Найдено {duplicates_count} дубликатов. Проставлен класс [300]")

    # --- Сохранение результата при необходимости ---
    if SAVE_STEP_5_RESULT and duplicates_count > 0:
        output_file = os.path.join(PROCESSED_DIR, f"step_5_duplicates_{TIMESTAMP}.xlsx")

        # --- Убираем таймзоны перед записью в Excel ---
        df_to_save = df.copy()
        if 'Дата создания' in df_to_save.columns:
            df_to_save['Дата создания'] = df_to_save['Дата создания'].apply(
                lambda x: x.replace(tzinfo=None) if pd.notna(x) and hasattr(x, 'tz') else x
            )

        try:
            df_to_save.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_300: {e}")

        # --- Сохраняем детали дубликатов ---
        details_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_duplicates_details_{TIMESTAMP}.xlsx")
        try:
            pd.DataFrame(detected_duplicates).to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Детали дубликатов сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить детали classifier_300: {e}")

        # --- Выводим статистику ---
        try:
            stats = class_statistics(df)
            logger.info(f"\n📊 [{step_number}] Статистика по всем классам после classifier_300:\n")
            logger.info(stats.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику — {e}")

    return df
