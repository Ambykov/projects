import json
import requests
from typing import List, Dict, Optional

def fetch_orcid_works(url: str) -> List[Dict]:
    """
    Загружает и извлекает данные из JSON-файла ORCID.

    Args:
        url: Прямая ссылка на JSON-файл.

    Returns:
        Список словарей с ключами: 'year', 'title', 'doi', 'eid'.
        Если идентификатор отсутствует, его значение будет None.
    """
    try:
        # 1. Загружаем JSON по URL
        print(f"Загрузка данных с {url}...")
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        print("Данные успешно загружены.")

        # 2. Получаем массив групп работ
        groups = data.get('groups', [])
        if not groups:
            print("Внимание: ключ 'groups' не найден или пуст.")
            return []

        results = []

        # 3. Проходим по каждой группе
        for group in groups:
            # Получаем массив works
            works = group.get('works', [])

            # Проходим по всем works в группе
            for work in works:
                # Извлекаем год публикации из publicationDate
                year = None
                pub_date = work.get('publicationDate', {})
                if pub_date and 'year' in pub_date:
                    year = pub_date['year']

                # Извлекаем название работы из title.value
                title = None
                title_obj = work.get('title', {})
                if title_obj and 'value' in title_obj:
                    title = title_obj['value']

                # Ищем DOI и EID в workExternalIdentifiers
                doi = None
                eid = None

                # Получаем внешние идентификаторы
                ext_ids = work.get('workExternalIdentifiers', [])

                for ext_id in ext_ids:
                    # Получаем тип идентификатора
                    id_type_obj = ext_id.get('externalIdentifierType', {})
                    id_type = id_type_obj.get('value') if isinstance(id_type_obj, dict) else id_type_obj

                    # Получаем значение идентификатора
                    id_value_obj = ext_id.get('externalIdentifierId', {})
                    ext_id_value = id_value_obj.get('value') if isinstance(id_value_obj, dict) else id_value_obj

                    if not ext_id_value:
                        continue

                    # Сохраняем в зависимости от типа
                    if id_type == 'doi' and not doi:
                        doi = ext_id_value
                    elif id_type == 'eid' and not eid:
                        eid = ext_id_value

                # Добавляем результат только если есть хотя бы год или название
                if year or title:
                    results.append({
                        'year': year,
                        'title': title,
                        'doi': doi,
                        'eid': eid
                    })

        print(f"Обработано записей: {len(results)}")
        return results

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке данных: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return []

def save_results_to_json(data: List[Dict], filename: str = "orcid_works_extracted.json"):
    """Сохраняет результат в JSON-файл с красивым форматированием."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в файл: {filename}")

def save_results_to_csv(data: List[Dict], filename: str = "orcid_works_extracted.csv"):
    """Сохраняет результат в CSV-файл."""
    try:
        import csv
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['year', 'title', 'doi', 'eid']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Результаты сохранены в CSV: {filename}")
    except ImportError:
        print("Модуль csv не найден, пропускаем сохранение в CSV")

# --- Основная часть программы ---
if __name__ == "__main__":
    # URL из задания
    target_url = "https://orcid.org/0000-0002-8520-7267/allWorks.json?sort=date&sortAsc=false"

    # Получаем данные
    extracted_data = fetch_orcid_works(target_url)

    # Выводим результаты
    if extracted_data:
        print(f"\nВсего записей: {len(extracted_data)}")
        print("\nПервые 10 записей для проверки:")
        print("-" * 80)
        for i, item in enumerate(extracted_data[:10]):
            print(f"{i+1:2}. Год: {item['year']}")
            print(f"    Название: {item['title']}")
            print(f"    DOI: {item['doi']}")
            print(f"    EID: {item['eid']}")
            print("-" * 80)

        # Сохраняем результаты
        save_results_to_json(extracted_data)
        save_results_to_csv(extracted_data)
    else:
        print("Не удалось извлечь данные.")