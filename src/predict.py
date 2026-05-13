import pandas as pd
import numpy as np
import psycopg2
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'pipeline'),
    'user': os.getenv('POSTGRES_USER', 'pipeline_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'pipeline_pass'),
}

TARGET_CURRENCIES = ['USD', 'EUR', 'RUB', 'CNY', 'GBP']


def get_latest_features(currency: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT fetch_date, rate / NULLIF(quantity, 0) as rate_per_unit
        FROM raw_exchange_rates
        WHERE currency_code = %s
        ORDER BY fetch_date DESC
        LIMIT 60
    """, conn, params=(currency,))
    conn.close()

    df = df.sort_values('fetch_date').reset_index(drop=True)
    df['fetch_date'] = pd.to_datetime(df['fetch_date'])
    df['day_of_week'] = df['fetch_date'].dt.dayofweek
    df['month'] = df['fetch_date'].dt.month
    df['quarter'] = df['fetch_date'].dt.quarter
    df['day_of_year'] = df['fetch_date'].dt.dayofyear
    df['lag_1'] = df['rate_per_unit'].shift(1)
    df['lag_3'] = df['rate_per_unit'].shift(3)
    df['lag_7'] = df['rate_per_unit'].shift(7)
    df['lag_14'] = df['rate_per_unit'].shift(14)
    df['rolling_mean_7'] = df['rate_per_unit'].rolling(7).mean()
    df['rolling_std_7'] = df['rate_per_unit'].rolling(7).std()
    df['rolling_mean_14'] = df['rate_per_unit'].rolling(14).mean()
    df['rolling_mean_30'] = df['rate_per_unit'].rolling(30).mean()
    df['daily_change'] = df['rate_per_unit'].diff()
    df['daily_change_pct'] = df['rate_per_unit'].pct_change()

    return df.dropna()


def predict_all():
    feature_cols = [
        'day_of_week', 'month', 'quarter', 'day_of_year',
        'lag_1', 'lag_3', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_mean_30',
        'daily_change', 'daily_change_pct'
    ]

    results = []
    pred_date = datetime.now().strftime('%Y-%m-%d')

    for currency in TARGET_CURRENCIES:
        model_path = f'models/{currency}_model.pkl'
        scaler_path = f'models/{currency}_scaler.pkl'

        if not os.path.exists(model_path):
            print(f"Model not found for {currency}")
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            df = get_latest_features(currency)
            if df.empty:
                continue

            last_row = df[feature_cols].iloc[[-1]]
            last_row_sc = scaler.transform(last_row)
            predicted = float(model.predict(last_row_sc)[0])
            actual = float(df['rate_per_unit'].iloc[-1])

            results.append({
                'currency': currency,
                'actual_rate': actual,
                'predicted_rate': predicted,
                'change': predicted - actual,
                'change_pct': (predicted - actual) / actual * 100
            })

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ml_predictions
                (prediction_date, currency_code, predicted_rate, actual_rate, model_name, mae)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (pred_date, currency, predicted, actual,
                  type(model).__name__, abs(predicted - actual)))
            conn.commit()
            cur.close()
            conn.close()

            print(f"{currency}: actual={actual:.2f}, predicted={predicted:.2f}, "
                  f"change={predicted-actual:+.2f} ({(predicted-actual)/actual*100:+.2f}%)")

        except Exception as e:
            print(f"Error for {currency}: {e}")

    return results


if __name__ == "__main__":
    predict_all()
