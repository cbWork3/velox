-- =============================================================================
-- SUM(int32) SVE 测试 - 数据库初始化脚本
-- =============================================================================
-- 首次运行: spark-sql --master yarn -f sum_int32_setup.sql
-- 然后运行: spark-sql --master yarn -f sum_int32_perf.sql
-- =============================================================================

CREATE DATABASE IF NOT EXISTS sum_int32_sve_test;
USE sum_int32_sve_test;

DROP TABLE IF EXISTS sum_int32_test_data;

CREATE TABLE sum_int32_test_data (
    group_key INT,
    value INT
)
USING PARQUET
LOCATION '/tmp/sum_int32_sve_test_data';

-- 生成约 5000 万行，1000 个分组（可调整 sequence 上限改变数据量）
INSERT OVERWRITE TABLE sum_int32_test_data
SELECT
    (id % 1000) AS group_key,
    (id % 10000) AS value
FROM (
    SELECT explode(sequence(1, 50000000)) AS id
);
