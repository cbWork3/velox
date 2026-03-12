-- =============================================================================
-- SUM(int32) SVE 性能测试 - 确保执行到 hashAggUpdateSVEWithCharForNormalInt32
-- =============================================================================
-- 执行方式:
--   1. 首次: spark-sql --master yarn -f sum_int32_setup.sql  (创建表并灌数)
--   2. 测试: spark-sql --master yarn -f sum_int32_perf.sql   (执行聚合)
--
-- 原理: pushdown 需 mayPushdown=true 且输入为 LazyVector
-- REPARTITION 在 Scan 与 Aggregation 间插入 Exchange，使 mayPushdown=false
-- =============================================================================

USE sum_int32_sve_test;

-- 方案 A [推荐]: REPARTITION 在 Scan 与 Agg 间插入 Exchange，mayPushdown=false
SELECT /*+ REPARTITION(200) */
    group_key,
    SUM(value) AS total
FROM sum_int32_test_data
GROUP BY group_key;

-- 若 REPARTITION 无效，可尝试方案 B（取消注释）:
-- SELECT group_key, SUM(v) AS total
-- FROM (SELECT group_key, value + 0 AS v FROM sum_int32_test_data) t
-- GROUP BY group_key;
