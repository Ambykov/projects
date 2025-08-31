# src/classifiers/classifier_700.py

import os
import pandas as pd
from tqdm import tqdm
import Levenshtein as lev


from config import logger, KEYWORDS_FILES, SAVE_STEP_10_RESULT, PROCESSED_DIR, TIMESTAMP
from utils import set_class
from utils import is_fuzzy_match_with_details as base_is_fuzzy_match_with_details
from decorators import pipeline_step



def has_ignored_class(classes):
    if isinstance(classes, list):
        return 999 in classes or 100 in classes
    return classes == 999 or classes == 100



# Словарь допустимых вариантов
ALLOWED_EXCEPTIONS = {
    'санкции':  ['секции', 'функции', 'акции', 'станции'],
    'натовец':  ['наконец'],
    'нато':     ['надо', 'нето', 'зато', 'наточен'],
    'сталин':   ['стало', 'стала', 'стал', 'стали', 'стадии', 'сдали', 'стакан', 'ставит', 'ставил', 'спали',
                 'сталь', 'устали', 'талон', 'встали', 'ставим', 'стати', 'столик'],
    'следком':  ['слишком', 'следом', 'легком', 'редком', 'сладкой'],
    'моди':     ['мои', 'моли', 'поди'],
    'путин':    ['пути'],
    'путен':    ['путем', 'пятен'],
    'минфин':   ['мини', 'минин'],
    'сенатор':  ['секатор', 'секаторы', 'сикатор', 'сенсор', 'сектор'],
    'макрон':   ['макаром', 'наклон', 'закон', 'мокрое', 'мокрая', 'макароны', 'микрон', 'марок', 'патрон'],
    'теракт':   ['теряет', 'терка', 'терки', 'терку', 'терке'],
    'протесты': ['протерта', 'потерты', 'провести', 'проценты'],
    'мафия':    ['магия'],
    'диверсия': ['версия'],
    'новак':    ['новая', 'новый', 'нова', 'новач'],
    'песков':   ['весов', 'леской', 'мешков', 'пешком', 'дисков', 'поисков', 'песке', 'пиков', 'кусков', 'песок'],
    'майдан':   ['задан', 'найден'],
    'бандит':   ['будит', 'банки'],
    'боевик':   ['бортик', 'болтик'],
    'реформа':  ['форма'],
    'бойкот':   ['мойкой', 'боком'],
    'захватчик': ['захватит'],
    'навальный': ['напольный', 'начальной', 'начальный'],
    'черненко': ['черенка', 'черенки', 'черенком'],
    'террор':   ['термос', 'термо'],
    'байден':  ['найден'],
    'фашизм':  ['вашим'],
    'преступление': ['поступление'],
    'маск':    ['маска', 'маски','маску','масок'],
    'брежнев': ['бережнее', 'бережно', 'брежно'],
    'ельцин': ['дельфин'],
    'галкин':  ['палки', 'гайки', 'галина'],
    'диверсия': ['доверия', 'версия'],
    'лавров':  ['коров', 'литров', 'ковров'],
    'убийца':  ['убила'],
    'смертник': ['смертный'],
    'уголовник': ['половник'],
    'судебная реформа': ['удобная форма'],
}


def is_fuzzy_match_with_details(review_text, key_phrases):
    # Просто вызываем исходную функцию из utils с дополнительным параметром
    return base_is_fuzzy_match_with_details(
        review_text,
        key_phrases,
    )


_key_phrases_cache = None


def load_phrases_from(filepath="key_words/keywords_700.py"):
    global _key_phrases_cache
    if _key_phrases_cache is not None:
        return _key_phrases_cache

    if not os.path.isfile(filepath):
        logger.error(f"❌ [10] Файл с ключевыми фразами не найден: {filepath}")
        raise FileNotFoundError(f"❌ [10] Файл с ключевыми фразами не найден: {filepath}")

    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("phrases_module", filepath)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'key_phrases'):
        key_phrases = getattr(module, 'key_phrases')
        logger.info("[10] Найдена переменная 'key_phrases' в модуле.")
    elif hasattr(module, 'KEY_WORDS_700'):
        key_phrases = getattr(module, 'KEY_WORDS_700')
        logger.info("[10] Найдена переменная 'KEY_WORDS_700' в модуле.")
    else:
        vars_in_file = dir(module)
        logger.error(f"❌ [10] В файле {filepath} отсутствует переменная 'key_phrases' или 'KEY_WORDS_700'. Найдены: {vars_in_file}")
        raise AttributeError(
            f"❌ [10] В файле {filepath} отсутствует переменная 'key_phrases' или 'KEY_WORDS_700'. Найдены: {vars_in_file}"
        )

    _key_phrases_cache = key_phrases
    return key_phrases


@pipeline_step(step_number=10, step_name="КЛАССИФИКАТОР [700]")
def classifier_700(df, stop_pipeline_flag=False, step_number=10):
    logger.info(f"🧠 [{step_number}] КЛАССИФИКАТОР [700]: старт")

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if 'Отзыв' not in df.columns or 'Класс' not in df.columns:
        logger.error(f"❌ [{step_number}] Колонки 'Отзыв' или 'Класс' отсутствуют в датафрейме")
        return df

    keyword_file_path = KEYWORDS_FILES.get(700)
    if not keyword_file_path or not os.path.exists(keyword_file_path):
        logger.error(f"❌ [{step_number}] Файл с ключевыми фразами для класса 700 не найден или не указан: {keyword_file_path}")
        return df

    try:
        key_phrases = load_phrases_from(keyword_file_path)
        logger.info(f"📌 [{step_number}] Загружено {len(key_phrases)} ключевых фраз политического контекста")
    except Exception as e:
        logger.error(f"❌ [{step_number}] Не удалось загрузить ключевые фразы: {e}")
        return df

    df_filtered = df[~df['Класс'].apply(has_ignored_class)].copy()

    if 'Примечание' not in df.columns:
        df['Примечание'] = ''

    detected_reviews = []
    count_found = 0

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="🧠 Инференс [700]"):
        if stop_pipeline_flag:
            logger.warning(f"🔚 [{step_number}] Классификация [700] прервана пользователем на строке {idx}")
            break

        review_text = str(row.get('Отзыв') or '').strip()
        if len(review_text) < 5:
            continue

        try:
            found, matches = is_fuzzy_match_with_details(review_text, key_phrases)

            # Отфильтровываем совпадения, исключая те, у которых найденное слово в списке исключений для ключевой фразы
            filtered_matches = []
            for orig, corr, source in matches:
                exceptions_for_source = ALLOWED_EXCEPTIONS.get(source, [])
                if corr in exceptions_for_source:
                    # Игнорируем совпадения из списка исключений (без логов)
                    continue
                else:
                    filtered_matches.append((orig, corr, source))

            if filtered_matches:
                count_found += 1
                df = set_class(df, idx, 700)

                note_text = "; ".join(
                    f"фраза: '{source}', найдено: '{corr}', в отзыве: '{orig}'"
                    for orig, corr, source in filtered_matches
                )
                old_note = df.at[idx, 'Примечание'] or ''
                df.at[idx, 'Примечание'] = f"{old_note}; политический контекст: {note_text}" if old_note else f"политический контекст: {note_text}"

                detected_reviews.append({
                    "Номер строки": idx,
                    "Текст отзыва": review_text,
                    "Детали совпадения": note_text,
                    "Продавец": row.get('Продавец', ''),
                    "Бренд": row.get('Название бренда', '')
                })
                logger.info(f"[{step_number}] Найден контекст на строке {idx}. Ключевая фраза: {note_text}.")

        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Ошибка на строке {idx}: {e}")

    logger.info(f"[{step_number}] Обработано строк: {len(df_filtered)}, с найденным политическим контекстом: {count_found}.")

    if SAVE_STEP_10_RESULT and detected_reviews:
        output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_classifier_700_{TIMESTAMP}.xlsx")
        try:
            df.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_700: {e}")

        details_df = pd.DataFrame(detected_reviews)
        details_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_classifier_700_details_{TIMESTAMP}.xlsx")
        try:
            details_df.to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Подробности по найденным отзывам сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить детали classifier_700: {e}")

        logger.info(f"✅ [{step_number}] Найдено {count_found} отзывов с политическим контекстом")
    else:
        logger.info(f"⚠️ [{step_number}] Не найдено ни одного отзыва с политическим контекстом")

    logger.info(f"✅ [{step_number}] КЛАССИФИКАТОР [700]: успешно выполнен")
    return df