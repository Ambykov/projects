from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
from datetime import datetime
import os
from loguru import logger
from tqdm import tqdm
import undetected_chromedriver as uc

# === Настройка логирования ===
os.makedirs("logs", exist_ok=True)
logger.add("logs/parser_{time}.log", rotation="daily")

# === Настройка браузера через undetected-chromedriver ===
def setup_browser():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

    # Убираем признаки автоматизации
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--enable-automation')
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')

    driver = uc.Chrome(options=options)
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
    })
    driver.execute_script("delete navigator.__proto__.webdriver;")
    return driver


# === Функция прокрутки для подгрузки всех товаров на одной странице ===
def scroll_page_to_load_all_products(page, delay=15):
    logger.info("[+] Начинаю прокрутку для подгрузки всех товаров на странице...")

    with tqdm(desc="Прокрутка страницы", unit="scroll") as pbar:
        prev_count = 0
        retries = 0
        max_retries = 5

        while True:
            page.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(delay)

            current_cards = page.find_elements(By.CSS_SELECTOR, "article.product-card")
            current_count = len(current_cards)

            pbar.update()
            pbar.set_postfix({"Товары": current_count})

            if current_count > prev_count:
                logger.debug(f"[+] Загружено {current_count} товаров")
                prev_count = current_count
                retries = 0
            else:
                retries += 1
                logger.warning(f"[!] Товары не подгружаются, попытка {retries}/{max_retries}")

            if retries >= max_retries:
                logger.info(f"[+] Больше нет новых товаров — останавливаю прокрутку ({current_count} шт.)")
                break


# === Получение общего числа товаров ===
def get_total_products_count(driver):
    try:
        goods_count_elem = driver.find_element(By.CSS_SELECTOR, ".goods-count span span")
        count_text = goods_count_elem.text.strip().replace(' ', '')
        total_products = int(count_text)
        logger.info(f"[+] Общее количество товаров: {total_products}")
        return total_products
    except Exception as e:
        logger.warning(f"[!] Не удалось получить количество товаров: {e}")
        return 100  # fallback значение


# === Парсинг товаров со страницы ===
def parse_page_products(driver, seller_id, page_num, seller_name):
    url = f"https://www.wildberries.ru/seller/{seller_id}?sort=popular&page={page_num}"
    logger.info(f"[+] Загрузка страницы {page_num}: {url}")
    driver.get(url)

    time.sleep(25)  # Даем время подгрузиться контенту

    # Прокрутить страницу до конца
    scroll_page_to_load_all_products(driver)

    # Поиск карточек товаров
    product_cards = driver.find_elements(By.CSS_SELECTOR, "article.product-card")
    logger.info(f"[+] Найдено товаров на странице {page_num}: {len(product_cards)}")

    parsed_data = []

    for i, card in enumerate(tqdm(product_cards, desc=f"Парсинг страницы {page_num}", unit="товар")):
        try:
            nm_id = card.get_attribute("data-nm-id")
            title_elem = card.find_element(By.CSS_SELECTOR, "a.product-card__link")
            name = title_elem.get_attribute("aria-label")
            href = title_elem.get_attribute("href")
        except NoSuchElementException as e:
            logger.warning(f"[!] Не найдены элементы в карточке {i + 1}: {e}")
            continue

        parsed_data.append({
            "ID продавца": seller_id,
            "Название продавца": seller_name,
            "артикул": nm_id,
            "название": name,
            "ссылка": href,
            "дата парсинга": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        time.sleep(1)  # Маленькая пауза для стабильности

    return parsed_data


# === Основная функция парсинга всех страниц ===
def parse_seller_products(seller_id):
    driver = setup_browser()

    # Открыть первую страницу
    first_page_url = f"https://www.wildberries.ru/seller/{seller_id}"
    driver.get(first_page_url)
    time.sleep(10)

    # Принять куки
    try:
        cookie_button = driver.find_element(By.CSS_SELECTOR, ".cookies__btn.btn-minor-md")
        cookie_button.click()
        logger.info("[+] Куки приняты")
    except Exception as e:
        logger.warning(f"[!] Не удалось найти кнопку принятия кук: {e}")

    # Получить имя продавца
    try:
        seller_name_elem = driver.find_element(By.CSS_SELECTOR, ".seller-details__title")
        seller_name = seller_name_elem.text.strip()
        logger.info(f"[+] Имя продавца: {seller_name}")
    except NoSuchElementException:
        logger.warning("[!] Не удалось получить имя продавца")
        seller_name = "Не найден"

    # Получить общее число товаров
    total_products = get_total_products_count(driver)
    items_per_page = 100  # Wildberries показывает по 100 товаров на странице
    pages_count = (total_products // items_per_page) + (1 if total_products % items_per_page > 0 else 0)
    logger.info(f"[+] Всего страниц: {pages_count} (по {items_per_page} товаров на странице)")

    all_products = []

    # Перебрать все страницы
    for page_num in range(1, pages_count + 1):
        if page_num % 5 == 1 and page_num > 1:
            logger.info("[+] Перезапуск браузера для обхода антибота")
            driver.quit()
            driver = setup_browser()
            time.sleep(20)

        products = parse_page_products(driver, seller_id, page_num, seller_name)
        all_products.extend(products)

        time.sleep(40)  # Пауза между страницами

    driver.quit()
    logger.info(f"[+] Всего спаршено товаров: {len(all_products)}")
    return all_products


# === Сохранение в Excel ===
def save_to_excel(data, seller_id):
    os.makedirs("results/sellers", exist_ok=True)
    df = pd.DataFrame(data, columns=[
        "ID продавца", "Название продавца", "артикул", "название", "ссылка", "дата парсинга"
    ])
    filename = f"results/sellers/wb_seller_{seller_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(filename, index=False)
    logger.success(f"[+] Товары сохранены в файл: {filename}")


# === Точка входа ===
if __name__ == "__main__":
    seller_id = input("Введите ID продавца: ").strip()

    if not seller_id.isdigit():
        logger.error("[!] ID продавца должен быть числом")
        exit()

    logger.info(f"[+] Парсинг товаров продавца {seller_id}")
    products = parse_seller_products(seller_id)

    if products:
        save_to_excel(products, seller_id)
    else:
        logger.warning("[!] Ни одного товара не найдено или произошла ошибка при парсинге")