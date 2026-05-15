WITH source AS (
    SELECT
        id,
        fetch_date,
        currency_code,
        currency_name,
        rate,
        quantity,
        created_at,
        CASE
            WHEN quantity > 0 THEN rate / quantity
            ELSE rate
        END AS rate_per_unit
    FROM {{ source('pipeline', 'raw_exchange_rates') }}
    WHERE
        rate IS NOT NULL
        AND rate > 0
        AND currency_code IS NOT NULL
        AND currency_code != ''
),

deduplicated AS (
    SELECT DISTINCT ON (fetch_date, currency_code)
        id,
        fetch_date,
        currency_code,
        currency_name,
        rate,
        quantity,
        rate_per_unit,
        created_at
    FROM source
    ORDER BY fetch_date, currency_code, created_at DESC
)

SELECT * FROM deduplicated
