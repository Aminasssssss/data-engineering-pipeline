import pandas as pd
import numpy as np
import psycopg2
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT fetch_date, currency_code, rate, quantity,
               rate / NULLIF(quantity, 0) as rate_per_unit
        FROM raw_exchange_rates
        WHERE currency_code = ANY(%s)
        ORDER BY fetch_date ASC
    """, conn, params=(TARGET_CURRENCIES,))
    conn.close()
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df['target'] = df['rate_per_unit'].shift(-1)
    return df.dropna()


def train_models(X_train, y_train, X_test, y_test):
    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        'Ridge': Ridge(alpha=1.0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'model': model, 'mae': mae, 'r2': r2}
        print(f"  {name}: MAE={mae:.4f}, R2={r2:.4f}")

    return results


def main():
    print("Loading data...")
    df = load_data()

    os.makedirs('models', exist_ok=True)

    feature_cols = [
        'day_of_week', 'month', 'quarter', 'day_of_year',
        'lag_1', 'lag_3', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_mean_30',
        'daily_change', 'daily_change_pct'
    ]

    all_results = {}

    for currency in TARGET_CURRENCIES:
        print(f"\nTraining models for {currency}...")
        currency_df = df[df['currency_code'] == currency].copy()

        if len(currency_df) < 30:
            print(f"  Not enough data for {currency}, skipping")
            continue

        currency_df = create_features(currency_df)

        if len(currency_df) < 20:
            continue

        X = currency_df[feature_cols]
        y = currency_df['target']

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        results = train_models(X_train_sc, y_train, X_test_sc, y_test)

        best_name = min(results, key=lambda k: results[k]['mae'])
        best_model = results[best_name]['model']

        joblib.dump(best_model, f'models/{currency}_model.pkl')
        joblib.dump(scaler, f'models/{currency}_scaler.pkl')

        all_results[currency] = {
            'best_model': best_name,
            'mae': results[best_name]['mae'],
            'r2': results[best_name]['r2']
        }

        print(f"  Best: {best_name} — MAE={results[best_name]['mae']:.4f}")

    print("\nAll models saved to models/")
    return all_results


if __name__ == "__main__":
    main()
