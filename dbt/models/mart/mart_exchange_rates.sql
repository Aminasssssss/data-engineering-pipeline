WITH staged AS (
    SELECT * FROM {{ ref('stg_exchange_rates') }}
),

with_analytics AS (
    SELECT
        fetch_date,
        currency_code,
        currency_name,
        rate_per_unit AS rate,

        AVG(rate_per_unit) OVER (
            PARTITION BY currency_code
            ORDER BY fetch_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_7d,

        AVG(rate_per_unit) OVER (
            PARTITION BY currency_code
            ORDER BY fetch_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_30d,

        rate_per_unit - LAG(rate_per_unit) OVER (
            PARTITION BY currency_code ORDER BY fetch_date
        ) AS daily_change,

        ROUND(
            (rate_per_unit - LAG(rate_unit) OVER (
                PARTITION BY currency_code ORDER BY fetch_date
            )) / NULLIF(LAG(rate_per_unit) OVER (
                PARTITION BY currency_code ORDER BY fetch_date
            ), 0) * 100, 4
        ) AS daily_change_pct,

        STDDEV(rate_per_unit) OVER (
            PARTITION BY currency_code
            ORDER BY fetch_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS volatility_7d,

        RANK() OVER (
            PARTITION BY fetch_date
            ORDER BY rate_per_unit DESC
        ) AS rate_rank

    FROM staged
)

SELECT * FROM with_analytics
ORDER BY fetch_date DESC, currency_code
