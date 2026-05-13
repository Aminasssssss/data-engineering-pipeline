# Data Engineering + ML Pipeline — Kazakhstan Exchange Rates

![Python](https://img.shields.io/badge/python-3.11-FFB6C1?style=flat-square&logo=python&logoColor=white)
![Airflow](https://img.shields.io/badge/airflow-2.8-FF69B4?style=flat-square&logo=apache-airflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/postgresql-15-FF69B4?style=flat-square&logo=postgresql&logoColor=white)
![Grafana](https://img.shields.io/badge/grafana-10.2-FFC0CB?style=flat-square&logo=grafana&logoColor=white)
![Docker](https://img.shields.io/badge/docker-compose-FFC0CB?style=flat-square&logo=docker&logoColor=white)

End-to-end data engineering and ML pipeline for forecasting Kazakhstan exchange rates using the National Bank of Kazakhstan open API.

## Architecture

```
National Bank of Kazakhstan API (free, no registration)
            |
    Airflow DAG (daily at 9:00)
            |
    PostgreSQL (raw_exchange_rates)
            |
    dbt (staging → mart models)
            |
    ML Model (GradientBoosting / RandomForest / Ridge)
            |
    PostgreSQL (ml_predictions)
            |
    Grafana Dashboard (real-time monitoring)
```

## Data Source

**National Bank of Kazakhstan (NBK) API** — official open API for daily exchange rates.
- URL: https://nationalbank.kz/rss/get_rates.cfm
- Free, no registration required
- Updates daily on weekdays
- Currencies: USD, EUR, RUB, CNY, GBP, JPY, CHF, KRW

## ML Models

Three regression models compete for each currency:
- GradientBoostingRegressor (200 estimators)
- RandomForestRegressor (200 estimators)
- Ridge Regression

Features: 14 engineered features including lag values (1, 3, 7, 14 days), rolling statistics (7, 14, 30 days), day-of-week, month, quarter, daily change.

Best model is selected per currency based on MAE on the test set.

## Results

| Currency | Best Model | MAE (KZT) | R² |
|----------|-----------|-----------|-----|
| USD/KZT | GradientBoosting | ~2.1 | 0.97 |
| EUR/KZT | GradientBoosting | ~3.4 | 0.96 |
| RUB/KZT | RandomForest | ~0.08 | 0.94 |
| CNY/KZT | GradientBoosting | ~0.9 | 0.95 |
| GBP/KZT | Ridge | ~4.2 | 0.95 |

## Quickstart

```bash
docker-compose up -d
```

Services:
- Airflow UI: http://localhost:8080 (admin/admin)
- Grafana: http://localhost:3000 (admin/admin)
- PostgreSQL: localhost:5432

Load historical data:
```bash
python src/fetch_data.py
```

Train models manually:
```bash
python src/train.py
```

Run predictions:
```bash
python src/predict.py
```

## Project Structure

```
data-engineering-pipeline/
├── airflow/
│   └── dags/
│       ├── collect_data.py   # DAG: fetch rates from NBK API daily
│       └── run_model.py      # DAG: train models and save predictions
├── dbt/
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_exchange_rates.sql
│   │   └── mart/
│   │       ├── mart_exchange_rates.sql
│   │       └── mart_predictions.sql
│   ├── sources.yml
│   └── dbt_project.yml
├── src/
│   ├── fetch_data.py   # Fetch historical data from NBK API
│   ├── train.py        # Train ML models per currency
│   └── predict.py      # Generate and save predictions
├── grafana/
│   ├── dashboards/
│   │   └── exchange_rates.json
│   └── provisioning/
├── notebooks/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Tech Stack

Python | Apache Airflow | dbt | PostgreSQL | scikit-learn | Grafana | Docker Compose

## Author

Zhumatayeva Amina — 2nd year Information Systems, KBTU
