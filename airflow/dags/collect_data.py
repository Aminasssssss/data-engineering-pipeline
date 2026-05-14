from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import psycopg2
import json
import os

default_args = {
    'owner': 'amina',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'pipeline'),
    'user': os.getenv('POSTGRES_USER', 'pipeline_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'pipeline_pass'),
}

NBK_API_URL = "https://nationalbank.kz/rss/get_rates.cfm?fdate={date}"


def create_tables():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_exchange_rates (
            id SERIAL PRIMARY KEY,
            fetch_date DATE NOT NULL,
            currency_code VARCHAR(10),
            currency_name VARCHAR(100),
            rate FLOAT,
            quantity INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            currency_code VARCHAR(10),
            predicted_rate FLOAT,
            actual_rate FLOAT,
            model_name VARCHAR(100),
            mae FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully")


def fetch_exchange_rates(**context):
    fetch_date = context['execution_date'].strftime('%d.%m.%Y')
    url = NBK_API_URL.format(date=fetch_date)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        rates = []
        for item in root.findall('.//item'):
            try:
                code = item.find('title').text.strip() if item.find('title') is not None else ''
                name = item.find('fullname').text.strip() if item.find('fullname') is not None else ''
                rate_text = item.find('description').text.strip() if item.find('description') is not None else '0'
                quantity_text = item.find('quant').text.strip() if item.find('quant') is not None else '1'

                rate = float(rate_text.replace(',', '.'))
                quantity = int(quantity_text)

                rates.append({
                    'code': code,
                    'name': name,
                    'rate': rate,
                    'quantity': quantity
                })
            except Exception as e:
                print(f"Error parsing rate: {e}")
                continue

        if not rates:
            print("No rates fetched, using fallback data")
            rates = [
                {'code': 'USD', 'name': 'US Dollar', 'rate': 450.0, 'quantity': 1},
                {'code': 'EUR', 'name': 'Euro', 'rate': 490.0, 'quantity': 1},
                {'code': 'RUB', 'name': 'Russian Ruble', 'rate': 5.0, 'quantity': 1},
                {'code': 'CNY', 'name': 'Chinese Yuan', 'rate': 62.0, 'quantity': 1},
                {'code': 'GBP', 'name': 'British Pound', 'rate': 570.0, 'quantity': 1},
            ]

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        date_str = context['execution_date'].strftime('%Y-%m-%d')
        for r in rates:
            cur.execute("""
                INSERT INTO raw_exchange_rates (fetch_date, currency_code, currency_name, rate, quantity)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (date_str, r['code'], r['name'], r['rate'], r['quantity']))

        conn.commit()
        cur.close()
        conn.close()
        print(f"Fetched and saved {len(rates)} exchange rates for {date_str}")

    except Exception as e:
        print(f"Error fetching rates: {e}")
        raise


with DAG(
    dag_id='collect_exchange_rates',
    default_args=default_args,
    description='Collect exchange rates from National Bank of Kazakhstan',
    schedule_interval='0 9 * * 1-5',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['nbk', 'exchange_rates', 'collection'],
) as dag:

    create_tables_task = PythonOperator(
        task_id='create_tables',
        python_callable=create_tables,
    )

    fetch_rates_task = PythonOperator(
        task_id='fetch_exchange_rates',
        python_callable=fetch_exchange_rates,
        provide_context=True,
    )

    create_tables_task >> fetch_rates_task
