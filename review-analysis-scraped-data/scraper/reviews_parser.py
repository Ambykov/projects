from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
import pandas as pd
from datetime import datetime, timedelta
import re
from loguru import logger
import os


# === Настройка логирования ===
os.makedirs("logs", exist_ok=True)
log_file = f"logs/reviews_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="daily")
logger.add(lambda msg: print(msg), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>", colorize=True)



# === Парсинг даты из отзыва ===
def parse_date(date_str):
    date_str = date_str.strip().lower()
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    # Убираем возможные кавычки
    if date_str.startswith("'"):
        date_str = date_str[1:]

    if date_str.startswith("сегодня"):
        time_match = re.search(r"\d{2}:\d{2}", date_str)
        if time_match:
            h, m = map(int, time_match.group().split(":"))
            return f"{today.year}-{today.month:02d}-{today.day:02d} {h:02d}:{m:02d}"
        return f"{today.year}-{today.month:02d}-{today.day:02d}"
    elif date_str.startswith("вчера"):
        time_match = re.search(r"\d{2}:\d{2}", date_str)
        if time_match:
            h, m = map(int, time_match.group().split(":"))
            return f"{yesterday.year}-{yesterday.month:02d}-{yesterday.day:02d} {h:02d}:{m:02d}"
        return f"{yesterday.year}-{yesterday.month:02d}-{yesterday.day:02d}"

    match_full_date = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4}),\s+(\d{2}:\d{2})", date_str)
    if match_full_date:
        day, month_name, year, time_str = match_full_date.groups()
        hour, minute = map(int, time_str.split(":"))
        month_map = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        month_num = month_map.get(month_name.lower(), None)
        if not month_num or month_num > 12:
            logger.warning(f"[!] Не удалось определить месяц в дате: {date_str}")
            return None
        return f"{year}-{month_num:02d}-{int(day):02d} {hour:02d}:{minute:02d}"

    match_short_date = re.match(r"(\d{1,2})\s+(\w+),\s+(\d{2}:\d{2})", date_str)
    if match_short_date:
        day, month_name, time_str = match_short_date.groups()
        hour, minute = map(int, time_str.split(":"))
        month_map = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        month_num = month_map.get(month_name.lower(), None)
        if not month_num or month_num > 12:
            logger.warning(f"[!] Не удалось определить месяц в дате: {date_str}")
            return None
        return f"{today.year}-{month_num:02d}-{int(day):02d} {hour:02d}:{minute:02d}"

    logger.warning(f"[!] Не удалось преобразовать дату: {date_str}")
    return None


# === Настройка браузера с обходом антибота ===
def setup_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-bot-detection")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)
    driver.execute_script("window.navigator = {webdriver: false};")
    return driver



# === Прокрутка страницы для загрузки отзывов с таймаутом ===
def scroll_to_load_all_reviews(driver, timeout=180, max_retries=5):
    logger.info("[ ] Начинаем прокрутку страницы")
    last_height = driver.execute_script("return document.body.scrollHeight")
    retries = 0
    start_time = time.time()

    while retries < max_retries:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(5, 10))
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            retries += 1
            logger.debug(f"[ ] Высота не изменилась, попытка {retries}/{max_retries}")
        else:
            retries = 0
            last_height = new_height
            logger.debug(f"[+] Новая высота: {last_height}")

        if time.time() - start_time > timeout:
            logger.warning(f"[!] Таймаут прокрутки ({timeout} сек.) достигнут")
            break

    logger.info("[+] Прокрутка завершена")



# === Функция парсинга отзывов по артикулу ===
def reviews_parser(product_id):
    logger.info(f"[+] Обработка артикула: {product_id}")

    driver = setup_browser()
    all_reviews = []

    try:
        # Переход на страницу товара
        product_url = f"https://www.wildberries.ru/catalog/{product_id}/detail.aspx"
        driver.get(product_url)
        time.sleep(random.uniform(5, 8))

        category_level_1 = category_level_2 = category_level_3 = brand = "Не указана"
        seller_name = "Не указан"

        # === Получаем цепочку категорий ===
        try:
            breadcrumb_items = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".breadcrumbs__link span[itemprop='name']"))
            )
            full_category_chain = [item.text.strip() for item in breadcrumb_items]
            if len(full_category_chain) >= 4:
                category_level_1 = full_category_chain[1] if len(full_category_chain) > 1 else "Не указана"
                category_level_2 = full_category_chain[2] if len(full_category_chain) > 2 else "Не указана"
                category_level_3 = full_category_chain[3] if len(full_category_chain) > 3 else "Не указана"
                brand = full_category_chain[-1] if len(full_category_chain) > 3 else "Не указан"
        except Exception as e:
            logger.warning(f"[!] Не удалось получить хлебные крошки: {e}")


        # === Переход на страницу отзывов ===
        feedback_url = f"https://www.wildberries.ru/catalog/{product_id}/feedbacks"
        driver.get(feedback_url)
        time.sleep(random.uniform(5, 8))

        # Принятие кук
        try:
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".cookies__btn.btn-minor-md"))
            )
            cookie_button.click()
            logger.info("[+] Куки приняты")
        except TimeoutException:
            logger.warning("[!] Не удалось найти кнопку принятия кук")

        # Прокрутка для подгрузки всех отзывов
        scroll_to_load_all_reviews(driver)

        review_cards = driver.find_elements(By.CSS_SELECTOR, ".comments__item.feedback.product-feedbacks__block-wrapper")
        logger.info(f"[+] Найдено {len(review_cards)} отзывов для артикула {product_id}")

        for card in review_cards:
            try:
                # Автор
                author_elem = card.find_element(By.CSS_SELECTOR, ".feedback__header")
                author = author_elem.text.strip() if author_elem else "Аноним"

                # Дата
                raw_date_elem = card.find_element(By.CSS_SELECTOR, ".feedback__date")
                raw_date = raw_date_elem.text.strip()
                date = parse_date(raw_date)

                # Оценка
                rating = "Нет оценки"
                try:
                    rating_elem = card.find_element(By.CSS_SELECTOR, ".feedback__rating.stars-line")
                    rating_class = rating_elem.get_attribute("class")
                    for part in rating_class.split():
                        if part.startswith("star") and len(part) > 4:
                            rating_str = ''.join(filter(str.isdigit, part))
                            if rating_str.isdigit():
                                rating = int(rating_str)
                                break
                except NoSuchElementException:
                    pass

                # Достоинства
                positive = ""
                try:
                    pos_elem = card.find_element(By.CSS_SELECTOR, ".feedback__text--item-pro .feedback__text--item-bold")
                    parent = pos_elem.find_element(By.XPATH, "./..")
                    full_text = parent.get_attribute("innerText").strip()
                    positive = full_text.replace("Достоинства:", "").strip()
                except NoSuchElementException:
                    pass

                # Недостатки
                negative = ""
                try:
                    neg_elem = card.find_element(By.CSS_SELECTOR, ".feedback__text--item-con .feedback__text--item-bold")
                    parent = neg_elem.find_element(By.XPATH, "./..")
                    full_text = parent.get_attribute("innerText").strip()
                    negative = full_text.replace("Недостатки:", "").strip()
                except NoSuchElementException:
                    pass

                # Комментарий
                comment_elems = card.find_elements(By.CSS_SELECTOR, "[itemprop='reviewBody'] .feedback__text--item")
                #comment_elems = card.find_elements(By.CSS_SELECTOR, ".feedback__text--item .feedback__text--item-bold")
                comment = ""
                if comment_elems:
                    full_text = comment_elems[0].text.strip()
                    comment = full_text.replace("Комментарий:", "").replace("Достоинства:", "").replace("Недостатки:", "").strip()

                # Убираем повторы в комментарии
                if positive and positive in comment:
                    comment = comment.replace(positive, "").strip()
                if negative and negative in comment:
                    comment = comment.replace(negative, "").strip()


                all_reviews.append({
                    "артикул": product_id,
                    "покупатель": author,
                    "дата": date,
                    "оценка": rating,
                    "достоинства": positive,
                    "недостатки": negative,
                    "комментарий": comment,
                    "категория_уровень_1": category_level_1,
                    "категория_уровень_2": category_level_2,
                    "категория_уровень_3": category_level_3,
                    "бренд": brand,
                    #"название_продавца": seller_name,
                })

            except Exception as e:
                logger.warning(f"[!] Ошибка при обработке отзыва: {e}")
                continue

    except Exception as e:
        logger.error(f"[!] Ошибка при парсинге артикула {product_id}: {e}")
    finally:
        try:
            driver.quit()
            logger.info(f"[+] Браузер закрыт для артикула {product_id}")
        except Exception as e:
            logger.warning(f"[!] Ошибка при закрытии браузера: {e}")

    if not all_reviews:
        logger.warning(f"[⚠️] Для артикула {product_id} нет отзывов или произошла ошибка")
        return pd.DataFrame(columns=[
            "артикул", "покупатель", "дата", "оценка", "достоинства",
            "недостатки", "комментарий", "категория_уровень_1", "категория_уровень_2",
            "категория_уровень_3", "бренд",
        ])

    df = pd.DataFrame(all_reviews, columns=[
        "артикул", "покупатель", "дата", "оценка", "достоинства",
        "недостатки", "комментарий", "категория_уровень_1", "категория_уровень_2",
        "категория_уровень_3", "бренд",
    ])

    # Преобразование числовых значений
    #df["артикул"] = pd.to_numeric(df["артикул"], errors='coerce').fillna(0).astype(int)
    #df["оценка"] = pd.to_numeric(df["оценка"], errors='coerce').fillna(0).astype(int)

    logger.success(f"[✅] Успешно спарсено {len(df)} отзывов для артикула {product_id}")
    return df