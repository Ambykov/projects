# src/classifiers/classifier_202.py

import os
import pandas as pd
from tqdm import tqdm

from utils import set_class, class_statistics, is_fuzzy_match_with_details
from decorators import pipeline_step
from config import (
    logger,
    SAVE_STEP_9_RESULT,
    PROCESSED_DIR,
    TIMESTAMP,
    FILTERED_OUT_CLASS,
    CLASS_100,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"  # Проверенная русскоязычная модель

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")


NEGATIVE_KEY_PHRASES = [
    "не рекомендую",
    "не советую",
    "не стоит",
    'не покупайте',
    'выброшенные деньги',
    'деньги на ветер',
    'скупой платит дважды',
    'не берите',

    "разочарован",
    'разочарована',
    "разочарование",
    "разочаровался",
    'разочаровала',
    'рассторен',
    'расстороена',
    'расстроил',
    'расстроила',
    'расстороило',
    'обидно',
    'досадно',
    'неприятно',
    'печально',
    'жаль',
    'обидно',
    'огорчена',
    'к сожалению',
    'но спустя',
    'негативные впечатления',
    'пожалела что купила',
    'расстроилась',
    'не смогла',
    'не смог',
    'ожидала другого',
    'не ожидала',
    'не довольна',
    'недовольна',
    'где гарантия',
    'не повезло',
    'обманули',
    'абсурдное требование',
    'не делала',
    'не делал',
    'не делали',
    'не заметили',
    'не понравилась',
    'не понравился',
    'не понравилось',
    'не понравились',

    "ужасный",
    'ужас',
    'ужасен',
    "худший",
    'худо',
    "плохо",
    'отвратительный',
    'отвратительно',
    "ненавижу",
    'кошмар',
    'лажа',
    'фигня',
    'фигней',
    'фигово',
    'гадость',
    'беда',
    'жуть',
    'отказ',
    'обман',
    'подло',
    'развод',
    'испорчено',
    'дефект',
    'неприятный сюрприз',
    'страшно открывать',
    "низкое качество",
    "высокая цена",
    'но внимание',
    'обманывает',

    "не понравился",
    'не понравилось',
    'не совпадает',
    'не соответствует',
    "не смог оценить",
    'отказался',
    'возврат',
    'вернул',
    'вернуть обратно',
    'придется возвращать',
    'забирать не буду',
    'не забрала',
    'не подошла',
    'больше брать не буду',
    'в футбол играли',
    'не проверил',
    'не понятно',

    "не привезли",
    "не доехал",
    'не доставили',
    'потеряли',
    'доставку задержали',
    'долгая доставка',
    'но доставка',
    'вместо нее',
    'не тот',
    'не того',

    'помяли',
    'мятая',
    'мятый',
    'мятое',
    'мятые',
    'вскрытая',
    'без фирменной упаковки',
    'не была в коробке',
    'без упаковки',
    'без коробки',
    'в непонятной упаковке',
    'не было',
    'небыло',

    'пришла без крышки',
    'пришел без',
    'пришли без',
    'прислали без',
    'в комплекте не было',
    'не комплект',
    'не в полной комплектации',
    'без ножа',
    'нет кнопки',
    'прислали какую то',
    'прислали какую-то',
    'нет ни какого',
    'не хватает',

    'забраковали',
    'взорвалась',
    'пригорела',
    'пригорает',
    'протекает',
    'вышел из строя',
    'не долго прослужила',
    'не работает',
    'работает через раз',
    'не поработал',
    'перестал работать',
    'не нажимаются',
    'повреждена',
    'дымится',
    'задымился',
    'сгорел',
    'начал протекать',
    'выгнулось',
    'не подходит',
    'не включается',
    'не включился',
    'не опускался',
    'сломалась',
    'перестала работать',
    'согнулась',
    'не закрывают',
    'не закрывается',
    'воняет',
    'барахлит',
    'не постирижешь',
    'не крутит',
    'не держит',
    'протекает',
    'не льется',
    'не держит',
    'перестал нагреваться',
    'не снимается',
    'заедает',
    'заржавел',
    'погнулась',
    'не оттирается',
    'оторвали',
    'вырубился',
    'не режет',
    'не нагревается',
    'не прокручивает',
    'виснет',
    'не выпрямляет',
    'появляется налет',
    'слышимость еле еле',
    'мутно стал показывать',
    'не варит',
    'не греет',
    'пластик плавится',
    'загорелась ошибка',
    'не вытягивает',
    'прилипает',
    'теряет',
    'не выдержал',
    'стал отключаться',
    'отвалилась',
    'перестала',
    'не прилегают',
    'затарахтел',
    'не приложили',
    'оплавилось',
    'оплавилась',
    'не подается',
    'появился треск',
    'завоняло',
    'завоняла',
    'не работало',
    'не работал',
    'не работала',
    'не работали',
    'не меняет',
    'не очищает',
    'не в состоянии',
    'быстро садится',
    'начал трескаться',

    'помятый',
    'помятая',
    'вмятина',
    'царапина',
    'поцарапанный',
    'скол',
    'сколами',
    'отколотая',
    'трещина',
    'отколотый',
    'разломан',
    'разбили',
    'битая',
    'слабенький',
    'подтекать',
    'пришел брак',
    'явный брак',
    'производственный брак',
    'бракованный',
    'это брак',
    'с браком',
    'бесполезная',
    'кривой',
    'косая',
    'гнутая',
    'плоховато',
    'не рабочий',
    'нерабочий',
    'нерабочем',
    'нерабочая',
    'сломанная',
    'сломаная',
    'вдребезги',
    'разбитый',
    'несоответствие',
    'не удобная',
    'неудобная',
    'не универсальная',
    'нет аккумулятора',
    'не качественная',
    'устал навсегда',
    'нарушена',
    'пара нет',

    'не оригинал',
    'подделка',
    'контрафакт',
    'паль',
    'не тефаль',
    'не tefal',
    'не fiskars',
]


#NEGATIVE_KEY_PHRASES = []

def contains_negative_phrase(text):
    text_lower = text.lower()
    found, matches = is_fuzzy_match_with_details(text_lower, NEGATIVE_KEY_PHRASES)
    return found, matches


def predict_sentiment(text_list):
    inputs = tokenizer(
        text_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


@pipeline_step(step_number=9, step_name="ГИБРИДНЫЙ ПОИСК ПОЛОЖИТЕЛЬНЫХ ОТЗЫВОВ → [202]")
def classifier_202(df, stop_pipeline_flag=False, positive_threshold=0.9, neutral_threshold=0.9, step_number=9):
    """
    Присваивает класс [202], если отзыв имеет вероятность положительной или нейтральной тональности выше порогов,
    и при этом не содержит негативных ключевых фраз (с учётом опечаток).
    """

    if stop_pipeline_flag:
        logger.warning(f"🔚 [{step_number}] Шаг прерван пользователем")
        return df

    if df is None or df.empty:
        logger.warning(f"🟡 [{step_number}] Входной DataFrame пустой или None")
        return df

    if "Отзыв" not in df.columns or "Класс" not in df.columns:
        logger.error(f"❌ [{step_number}] Входной DataFrame должен содержать колонки 'Отзыв' и 'Класс'")
        return df

    if "Примечание" not in df.columns:
        df["Примечание"] = ""

    # Создаём / обнуляем колонки вероятностей
    if 'prob_positive_202' not in df.columns:
        df['prob_positive_202'] = pd.NA
    if 'prob_neutral_202' not in df.columns:
        df['prob_neutral_202'] = pd.NA

    def has_ignored_class(classes):
        if isinstance(classes, list):
            return FILTERED_OUT_CLASS in classes or CLASS_100 in classes
        return classes == FILTERED_OUT_CLASS or classes == CLASS_100

    df_to_process = df[~df["Класс"].apply(has_ignored_class)].copy()
    if df_to_process.empty:
        logger.info(f"ℹ️ [{step_number}] Нет отзывов для обработки после фильтра по классам")
        return df

    found_count = 0
    detected_positive_reviews = []

    batch_size = 32
    reviews = df_to_process["Отзыв"].astype(str).tolist()
    indices = df_to_process.index.tolist()

    for start_idx in tqdm(
        range(0, len(reviews), batch_size),
        desc=f"🔎 [{step_number}] Гибридный поиск положительных отзывов [202]",
    ):
        batch_reviews = reviews[start_idx : start_idx + batch_size]
        batch_indices = indices[start_idx : start_idx + batch_size]
        probs = predict_sentiment(batch_reviews)

        for idx_in_batch, prob in enumerate(probs):
            negative_prob = prob[0]
            neutral_prob = prob[1]
            positive_prob = prob[2]

            idx_df = batch_indices[idx_in_batch]
            text = df.at[idx_df, "Отзыв"]

            found_neg, neg_matches = contains_negative_phrase(text)

            if (positive_prob >= positive_threshold or neutral_prob >= neutral_threshold) and not found_neg:
                df = set_class(df, idx_df, code=202)

                # Сохраняем вероятности в DataFrame для дальнейшего использования
                df.at[idx_df, 'prob_positive_202'] = positive_prob
                df.at[idx_df, 'prob_neutral_202'] = neutral_prob

                old_note = df.at[idx_df, "Примечание"]
                new_note = (
                    f"{old_note} | положительный/нейтральный отзыв "
                    f"(вероятность pos={positive_prob:.2f}, neu={neutral_prob:.2f})"
                ) if old_note else f"положительный/нейтральный отзыв (вероятность pos={positive_prob:.2f}, neu={neutral_prob:.2f})"
                df.at[idx_df, "Примечание"] = new_note

                found_count += 1
                detected_positive_reviews.append(
                    {
                        "Номер строки": idx_df,
                        "Отзыв": text,
                        "Продавец": df.at[idx_df, "Продавец"] if "Продавец" in df.columns else "",
                        "Вероятность положительной тональности": positive_prob,
                        "Вероятность нейтральной тональности": neutral_prob,
                        "Найдено негативных ключевых фраз": neg_matches,
                    }
                )

    # Заменяем все значения NaN (pd.NA) в колонках с вероятностями на пустые строки
    for col in ['prob_positive_202', 'prob_neutral_202']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '' if pd.isna(x) else x)

    logger.info(
        f"✅ [{step_number}] Найдено {found_count} положительных или нейтральных отзывов (гибридный метод). Проставлен класс [202]"
    )

    if SAVE_STEP_9_RESULT and found_count > 0:
        output_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_positive_reviews_{TIMESTAMP}.xlsx")
        try:
            df.to_excel(output_file, index=False)
            logger.info(f"📌 [{step_number}] Результат сохранён в: {output_file}")
        except Exception as e:
            logger.warning(f"⚠️ [{step_number}] Не удалось сохранить результат classifier_202: {e}")

        details_file = os.path.join(PROCESSED_DIR, f"step_{step_number}_positive_reviews_details_{TIMESTAMP}.xlsx")
        try:
            pd.DataFrame(detected_positive_reviews).to_excel(details_file, index=False)
            logger.info(f"📁 [{step_number}] Детали положительных отзывов сохранены в: {details_file}")
        except Exception as e:
            logger.warning(f"⚠️[{step_number}] Не удалось сохранить детали classifier_202: {e}")

        try:
            stats = class_statistics(df)
            logger.info(f"\n📊 [{step_number}] Статистика по всем классам после classifier_202:\n")
            logger.info(stats.to_string(index=False))
        except Exception as e:
            logger.warning(f"⚠️[{step_number}] Не удалось вывести статистику — {e}")

    return df
