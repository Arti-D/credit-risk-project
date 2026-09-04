-- общ баланс
SELECT
    serious_dlqin2yrs,
    COUNT(*) AS cnt,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM credit_data), 2) AS pct
FROM credit_data
GROUP BY serious_dlqin2yrs;

-- ср возраст + доход по группам (дефолт / без дефолта)
SELECT
    serious_dlqin2yrs,
    ROUND(AVG(age), 1)            AS avg_age,
    ROUND(AVG(monthly_income), 1) AS avg_income,
    ROUND(AVG(debt_ratio), 3)     AS avg_debt_ratio
FROM credit_data
GROUP BY serious_dlqin2yrs;

-- дефолты по возрастным группам
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END AS age_group,
    COUNT(*)                                             AS total,
    SUM(serious_dlqin2yrs)                                AS defaults,
    ROUND(100.0 * SUM(serious_dlqin2yrs) / COUNT(*), 2)   AS default_rate_pct
FROM credit_data
GROUP BY age_group
ORDER BY age_group;

-- дефолты по открытым кредитным линиям
SELECT
    CASE
        WHEN number_of_open_credit_lines_and_loans <= 5 THEN '0-5'
        WHEN number_of_open_credit_lines_and_loans <= 10 THEN '6-10'
        WHEN number_of_open_credit_lines_and_loans <= 15 THEN '11-15'
        ELSE '16+'
    END AS lines_group,
    COUNT(*)                                            AS total,
    ROUND(100.0 * SUM(serious_dlqin2yrs) / COUNT(*), 2)  AS default_rate_pct
FROM credit_data
GROUP BY lines_group
ORDER BY lines_group;

-- с пропущенным доходом
SELECT COUNT(*) AS missing_income_count
FROM credit_data
WHERE monthly_income IS NULL;

-- выбросы
SELECT COUNT(*) AS outliers_count
FROM credit_data
WHERE age < 18 OR debt_ratio > 10;
