import asyncio
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import logging
from check_server_connection import check_server_connection
from config import TIMEZONE_OFFSET


logger = logging.getLogger(__name__)


def export_data_by_tag(
    tag_number,
    db_host="emcable-and-amai.cyberb2b.ru",
    db_port=15432,
    db_name="mscada_db",
    db_user="student",
    db_password="5SxdeChZ"
):
    """
    Выгружает данные из БД за последние 10 секунд
    """

    logger.info(f"🔍 Запрашиваем данные для тега {tag_number} за последние 10 секунд")

    if not check_server_connection(db_host, db_port):
        raise ConnectionError(f"❌ Сервер {db_host}:{db_port} недоступен")

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=10)

        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        # Windows Filetime: количество 100-нс интервалов с 1601-01-01
        start_ts = int((start_time.timestamp() + 11644473600) * 10_000_000)
        end_ts = int((end_time.timestamp() + 11644473600) * 10_000_000)

        with conn.cursor(name=f'server_side_cursor_{tag_number}') as cursor:
            query = """
            SELECT source_time, value
            FROM data_raw
            WHERE archive_itemid = %s AND layer IN (0, 1)
              AND source_time BETWEEN %s AND %s
            ORDER BY source_time;
            """
            cursor.execute(query, (tag_number, start_ts, end_ts))
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=['source_time', 'value'])

        # Конвертация Windows Filetime -> Unix timestamp -> datetime (UTC)
        df['source_time'] = pd.to_datetime(
            (df['source_time'] / 10_000_000) - 11644473600,
            unit='s',
            utc=True
        )

        # Применяем смещение из config.TIMEZONE_OFFSET
        df['source_time'] = df['source_time'] + pd.Timedelta(hours=TIMEZONE_OFFSET)

        df.rename(columns={'value': f'value_{tag_number}'}, inplace=True)

        return df[['source_time', f'value_{tag_number}']].copy()

    except Exception as e:
        logger.error(f"❌ Ошибка при выгрузке данных для тега {tag_number}: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()