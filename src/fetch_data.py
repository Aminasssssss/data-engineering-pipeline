import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import os
import time

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', '127.0.0.1'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'pipeline'),
    'user': os.getenv('POSTGRES_USER', 'pipeline_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'pipeline_pass'),
}

NBK_API_URL = "https://nationalbank.kz/rss/get_rates.cfm?fdate={date}"
TARGET_CURRENCIES = ['USD', 'EUR', 'RUB', 'CNY', 'GBP', 'JPY', 'CHF', 'KRW']


def fetch_rates_for_date(date: datetime) -> list:
    date_str = date.strftime('%d.%m.%Y')
    url = NBK_API_URL.format(date=date_str)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        rates = []

        for item in root.findall('.//item'):
            try:
                code = item.find('title').text.strip()
                name = item.find('fullname').text.strip()
                rate = float(item.find('description').text.strip().replace(',', '.'))
                quantity = int(item.find('quant').text.strip())

                if code in TARGET_CURRENCIES:
                    rates.append({
                        'fetch_date': date.strftime('%Y-%m-%d'),
                        'currency_code': code,
                        'currency_name': name,
                        'rate': rate,
                        'quantity': quantity
                    })
            except Exception:
                continue

        return rates

    except Exception as e:
        print(f"Error fetching rates for {date_str}: {e}")
        return []


def save_to_db(rates: list):
    if not rates:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for r in rates:
        cur.execute("""
            INSERT INTO raw_exchange_rates
            (fetch_date, currency_code, currency_name, rate, quantity)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (r['fetch_date'], r['currency_code'], r['currency_name'],
              r['rate'], r['quantity']))

    conn.commit()
    cur.close()
    conn.close()


def fetch_historical_data(days: int = 365):
    print(f"Fetching {days} days of historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current = start_date
    total_saved = 0

    while current <= end_date:
        if current.weekday() < 5:
            rates = fetch_rates_for_date(current)
            if rates:
                save_to_db(rates)
                total_saved += len(rates)
                print(f"Saved {len(rates)} rates for {current.strftime('%Y-%m-%d')}")
            time.sleep(0.5)
        current += timedelta(days=1)

    print(f"Total saved: {total_saved} records")


if __name__ == "__main__":
    fetch_historical_data(days=365)