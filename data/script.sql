-- СОЗДАНИЕ ТАБЛИЦЫ

CREATE TABLE startup_funding (
    -- id SERIAL PRIMARY KEY,
    startup_stage TEXT NOT NULL,
    industry TEXT NOT NULL,
    region TEXT NOT NULL,
    requested_amount NUMERIC(18, 6),
    pre_money_valuation NUMERIC(18, 6),
    team_size INTEGER,
    founders_experience_years INTEGER,
    annual_revenue NUMERIC(18, 6),
    market_size_estimate NUMERIC(18, 6),
    investment_amount NUMERIC(18, 6)
);

-- ОБЩИЙ ОБЗОР

-- Обзор
SELECT * 
FROM startup_funding 
LIMIT 5;

-- Количество существующих заявок
SELECT COUNT(*) AS total_startups 
FROM startup_funding;

-- Количество стартапов по индустриям
SELECT 
    industry, 
    COUNT(*) AS industry_count
FROM startup_funding 
GROUP BY industry;

-- Количество стартапов по стадиям
SELECT 
    COUNT(startup_stage) AS startups_count_by_stage
FROM startup_funding 
GROUP BY startup_stage;

-- АНАЛИЗ ИНВЕСТИЦИЙ И ОДОБРЕНИЙ

-- Соотношение одобренных и неодобренных стартапов по индустриям
SELECT
    industry,
    SUM(CASE WHEN investment_amount <> 0 THEN 1 ELSE 0 END) AS sponsored,
    SUM(CASE WHEN investment_amount = 0 THEN 1 ELSE 0 END) AS rejected,
    SUM(CASE WHEN investment_amount <> 0 THEN 1 ELSE 0 END)::float
    /
    NULLIF(SUM(CASE WHEN investment_amount = 0 THEN 1 ELSE 0 END), 0) AS ratio
FROM startup_funding
GROUP BY industry 
ORDER BY ratio DESC;

-- Соотношение одобренных и неодобренных стартапов по регионам
SELECT
    region,
    SUM(CASE WHEN investment_amount <> 0 THEN 1 ELSE 0 END) AS sponsored,
    SUM(CASE WHEN investment_amount = 0 THEN 1 ELSE 0 END) AS rejected,
    SUM(CASE WHEN investment_amount <> 0 THEN 1 ELSE 0 END)::float
    /
    NULLIF(SUM(CASE WHEN investment_amount = 0 THEN 1 ELSE 0 END), 0) AS ratio
FROM startup_funding 
GROUP BY region 
ORDER BY ratio DESC;

-- Индустрии по количеству отказов (по убыванию)
SELECT 
    industry, 
    COUNT(industry)
FROM startup_funding
WHERE investment_amount = 0
GROUP BY industry
ORDER BY COUNT(industry) DESC;

-- Самые инвестируемые стадии
SELECT 
    startup_stage, 
    AVG(investment_amount) AS average_investment_amount
FROM startup_funding
WHERE investment_amount <> 0
GROUP BY startup_stage
ORDER BY average_investment_amount DESC;

-- Топ-3 индустрии по среднему инвестиционному чеку
SELECT 
    industry, 
    ROUND(AVG(investment_amount), 2) AS average_investment_amount
FROM startup_funding
GROUP BY industry 
ORDER BY average_investment_amount DESC
LIMIT 3;

-- Индустрии с высшим объёмом инвестиций по датасету
SELECT 
    industry, 
    AVG(investment_amount)
FROM startup_funding
GROUP BY industry
HAVING AVG(investment_amount) > (
    SELECT AVG(investment_amount) 
    FROM startup_funding
);

-- ФИНАНСОВЫЙ АНАЛИЗ

-- Запрашиваемые суммы инвестиций
SELECT 
    MIN(requested_amount) AS mininimal_requested_amount,
    MAX(requested_amount) AS maximal_requested_amount,
    AVG(requested_amount) AS average_requested_amount
FROM startup_funding;

-- Запрашиваемые инвестиции и объёмы инвестиций на каждую индустрию
SELECT 
    industry, 
    ROUND(AVG(requested_amount)) AS average_requested_amount,
    ROUND(AVG(investment_amount)) AS average_investment_amount
FROM startup_funding
GROUP BY industry
ORDER BY average_investment_amount DESC;

-- Топ стартапов по соотношению запрошенной суммы к предварительной стоимости
SELECT 
    *,
    requested_amount, 
    pre_money_valuation,
    requested_amount / pre_money_valuation AS ratio
FROM startup_funding
ORDER BY ratio DESC
LIMIT 3;

-- Переоценённые стартапы
SELECT 
    COUNT(*) AS overpriced_startups
FROM startup_funding
WHERE pre_money_valuation > requested_amount * 3;

-- АНАЛИЗ РЫНКА И РЕГИОНОВ

-- Средний объём рынка по регионам
SELECT 
    region, 
    AVG(market_size_estimate) AS average_market_size_estimate
FROM startup_funding
GROUP BY region
ORDER BY average_market_size_estimate DESC;

-- Регионы по approval rate
SELECT region, COUNT(*) as approved_startups FROM startup_funding
	WHERE investment_amount <> 0
	GROUP BY region
	ORDER BY approved_startups DESC

-- АНАЛИЗ КОМАНД И ОСНОВАТЕЛЕЙ

-- Средний опыт владельцев стартапов в зависимости от стадии
SELECT 
    startup_stage, 
    ROUND(AVG(founders_experience_years)) AS average_founder_experience_years
FROM startup_funding
GROUP BY startup_stage
ORDER BY startup_stage ASC;

-- Размер команды по индустриям
SELECT
	industry,
	ROUND(AVG(team_size)) as average_team_size,
	MAX(team_size) as max_team_size,
	MIN(team_size) as min_team_size
FROM startup_funding
GROUP by industry

-- АНАЛИЗ ВЫРУЧКИ

-- Стартапы без выручки
SELECT 
    COUNT(*) AS no_revenue_startups
FROM startup_funding
WHERE annual_revenue = 0;