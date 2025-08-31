# src/classifiers/classifier_200.py

import os
import re
import pandas as pd
from tqdm import tqdm

from decorators import pipeline_step
from utils import set_class, class_statistics
from config import logger, SAVE_STEP_4_RESULT, PROCESSED_DIR, TIMESTAMP, FILTERED_OUT_CLASS, CLASS_100

# Регулярные выражения для ссылок, телеграм, соцсетей
URL_PATTERNS = [
    r"https?://(?:www\.)?\S+",
    r"www\.\S+",
    r"\S+\.(?:com|ru|net|org|info|biz|gov|edu|io|co|me)(?:/\S*)?",
]

SOCIAL_MEDIA_PATTERNS = [
    r"t\.me/\S+",
    r"telegram\.me/\S+",
    r"vk\.com/\S+",
    r"facebook\.com/\S+",
    r"instagram\.com/\S+",
    r"ok\.ru/\S+",
    r"twitter\.com/\S+",
    r"youtube\.com/\S+",
]

KEYWORD_PATTERNS = [
    r"\bинтернет-ресурс\b",
]

# Ключевые фразы для расширенного поиска (с некаптурирующими группами)
spam_regex = [
    # 🎯 Явные рекламные и промо-фразы
    r"\bскидки до\b", r"\bкупон на скидку\b", r"\bскидочный купон\b", r"\bсупер цена\b", r"\bакция\b",
    r"\bраспродажа\b", r"\bраспродажа товаров\b", r"\bуспей купить\b", r"\bвыгодное предложение\b",
    r"\bограниченное предложение\b", r"\bполучить подарок\b", r"\bполучить подарки\b",
    r"\bполучить бесплатно\b", r"\bприз\b", r"\bвыиграй\b", r"\bконкурс\b", r"\bрозыгрыш\b",

    # 🔗 Ссылки и переходы
    r"\bпо промокоду\b", r"\bпереходи по ссылке\b", r"\bпереходите по ссылке\b",
    r"\bссылка в профиле\b", r"\bссылка ниже\b", r"\bна нашем сайте\b", r"\bищите в гугле\b", r"\bищите в яндексе\b",
    r"(?:wa\.me|t\.me|вотсап|\bтелега\b|\bвайбер\b)", r"(?:реферальная|партнёрская)\s+ссылка",
    r"(?:заработай|получи)\s+(?:бонус|скидку|кэшбэк)\s+по\s+ссылке",

    # 💬 Призывы к связям / контактам
    r"\bпиши в личку\b", r"\bнапиши в личку\b", r"\bнаписать в личку\b", r"\bпишите в лс\b", r"\bв директ\b",
    r"\bличные сообщения\b", r"(?:пишите|свяжитесь)\s+в\s+(?:личку|директ|личные\s+сообщения)",
    r"\bмы вам перезвоним\b", r"\bоставьте номер\b", r"\bсвяжемся с вами\b",
    r"(?:вступай|вступите)\s+в\s+(?:чат|группу)",

    # 📱 Мессенджеры и соцсети
    r"\bтелеграм\b", r"\btg\b", r"\btelegram\b", r"\bвайбер\b", r"\bviber\b", r"\bватсап\b", r"\bwhatsapp\b",
]

ALL_PATTERNS = URL_PATTERNS + SOCIAL_MEDIA_PATTERNS + KEYWORD_PATTERNS
COMPILED_PATTERN = re.compile("|".join(ALL_PATTERNS), flags=re.IGNORECASE)
SPAM_PATTERN = re.compile("|".join(spam_regex), flags=re.IGNORECASE)


def extract_matches(findall_result):
    """
    Обрабатывает результат re.findall, возвращая список строк.
    Если шаблон содержит группы, findall возвращает кортежи.
    Берёт первый непустой элемент из каждого кортежа.
    """
    result = []
    for item in findall_result:
        if isinstance(item, tuple):
            for subitem in item:
                if subitem:
                    result.append(subitem)
                    break
        elif isinstance(item, str):
            result.append(item)
    return result


@pipeline_step(step_number=4, step_name="КЛАССИФИКАТОР [200] — Поиск ссылок и ключевых фраз")
def classifier_200(df, stop_pipeline_flag=False, step_number=4):
    logger.info(f"🔍 [{step_number}] КЛАССИФИКАТОР [200]: старт – поиск ссылок и ключевых фраз")

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной датафрейм пустой или None")
        return df

    if 'Класс' not in df.columns:
        logger.error(f"❌ [{step_number}] Отсутствует колонка 'Класс' в датафрейме")
        return df

    def has_ignored_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes or CLASS_100 in classes
        return classes == FILTERED_OUT_CLASS or classes == CLASS_100

    df_filtered = df[~df['Класс'].apply(has_ignored_class)]

    found_count = 0
    detected_reviews = []

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"🔍 [{step_number}] Поиск ссылок и ключевых фраз [200]"):
        review_text = str(row.get('Отзыв', '')).lower()

        raw_links = COMPILED_PATTERN.findall(review_text)
        raw_spam = SPAM_PATTERN.findall(review_text)

        matches_links = set(extract_matches(raw_links))
        matches_spam = set(extract_matches(raw_spam))

        all_matches = matches_links | matches_spam

        if all_matches:
            df = set_class(df, idx, 200)
            found_count += 1

            old_note = df.at[idx, 'Примечание'] if 'Примечание' in df.columns else ''
            found_str = ", ".join(sorted(all_matches))
            new_note = f"{old_note} | найдено: {found_str}" if old_note else f"найдено: {found_str}"
            df.at[idx, 'Примечание'] = new_note

            detected_reviews.append({
                'Номер строки': idx,
                'Отзыв': row.get('Отзыв', ''),
                'Найдены ссылки/фразы': found_str
            })

    logger.info(f"📌 [{step_number}] Найдено {found_count} отзывов с ссылками или ключевыми фразами. Проставлен класс [200]")

    if SAVE_STEP_4_RESULT and found_count > 0:
        try:
            output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_classifier_200_{TIMESTAMP}.xlsx")
            df.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результаты classifier_200: {e}")

        try:
            details_file = os.path.join(PROCESSED_DIR, f"step_4_classifier_200_details_{TIMESTAMP}.xlsx")
            pd.DataFrame(detected_reviews).to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Детали найденных отзывов сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить детали classifier_200: {e}")

        try:
            stats = class_statistics(df)
            logger.info(f"\n📊 [{step_number}] Статистика по классам после classifier_200:\n%s", stats.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось вывести статистику classifier_200: {e}")

    return df
