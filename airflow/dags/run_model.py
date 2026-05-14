from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, '/opt/airflow/src')

default_args = {
    'owner': 'amina',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'email_on_failure': False,
}

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'pipeline'),
    'user': os.getenv('POSTGRES_USER', 'pipeline_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'pipeline_pass'),
}


def run_ml_pipeline(**context):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT fetch_date, currency_code, rate
        FROM raw_exchange_rates
        WHERE currency_code IN ('USD', 'EUR', 'RUB', 'CNY', 'GBP')
        ORDER BY fetch_date ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty or len(df) < 30:
        print("Not enough data for training")
        return

    df['fetch_date'] = pd.to_datetime(df['fetch_date'])

    for currency in df['currency_code'].unique():
        try:
            currency_df = df[df['currency_code'] == currency].copy()
            currency_df = currency_df.sort_values('fetch_date').reset_index(drop=True)

            if len(currency_df) < 20:
                continue

            currency_df['day_of_week'] = currency_df['fetch_date'].dt.dayofweek
            currency_df['month'] = currency_df['fetch_date'].dt.month
            currency_df['lag_1'] = currency_df['rate'].shift(1)
            currency_df['lag_3'] = currency_df['rate'].shift(3)
            currency_df['lag_7'] = currency_df['rate'].shift(7)
            currency_df['rolling_mean_7'] = currency_df['rate'].rolling(7).mean()
            currency_df['rolling_std_7'] = currency_df['rate'].rolling(7).std()
            currency_df['rolling_mean_14'] = currency_df['rate'].rolling(14).mean()
            currency_df['target'] = currency_df['rate'].shift(-1)

            currency_df = currency_df.dropna()

            if len(currency_df) < 15:
                continue

            feature_cols = ['day_of_week', 'month', 'lag_1', 'lag_3', 'lag_7',
                           'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14']

            X = currency_df[feature_cols]
            y = currency_df['target']

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            models = {
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Ridge': Ridge(alpha=1.0),
            }

            best_model = None
            best_mae = float('inf')
            best_name = ''

            for name, model in models.items():
                model.fit(X_train_sc, y_train)
                preds = model.predict(X_test_sc)
                mae = mean_absolute_error(y_test, preds)
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_name = name

            last_row = X.iloc[[-1]]
            last_row_sc = scaler.transform(last_row)
            predicted_rate = float(best_model.predict(last_row_sc)[0])
            actual_rate = float(currency_df['rate'].iloc[-1])

            pred_date = context['execution_date'].strftime('%Y-%m-%d')

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ml_predictions
                (prediction_date, currency_code, predicted_rate, actual_rate, model_name, mae)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (pred_date, currency, predicted_rate, actual_rate, best_name, best_mae))
            conn.commit()
            cur.close()
            conn.close()

            print(f"{currency}: predicted={predicted_rate:.2f}, actual={actual_rate:.2f}, MAE={best_mae:.4f}, model={best_name}")

        except Exception as e:
            print(f"Error for {currency}: {e}")
            continue


with DAG(
    dag_id='run_ml_predictions',
    default_args=default_args,
    description='Train ML models and predict exchange rates',
    schedule_interval='0 10 * * 1-5',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['ml', 'predictions', 'exchange_rates'],
) as dag:

    ml_task = PythonOperator(
        task_id='run_ml_pipeline',
        python_callable=run_ml_pipeline,
        provide_context=True,
    )
