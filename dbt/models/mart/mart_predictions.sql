WITH predictions AS (
    SELECT * FROM {{ source('pipeline', 'ml_predictions') }}
),

exchange_rates AS (
    SELECT * FROM {{ ref('stg_exchange_rates') }}
),

joined AS (
    SELECT
        p.prediction_date,
        p.currency_code,
        p.predicted_rate,
        p.actual_rate,
        p.model_name,
        p.mae,
        ABS(p.predicted_rate - p.actual_rate) AS absolute_error,
        ROUND(
            ABS(p.predicted_rate - p.actual_rate) /
            NULLIF(p.actual_rate, 0) * 100, 4
        ) AS mape
    FROM predictions p
)

SELECT * FROM joined
ORDER BY prediction_date DESC, currency_code
