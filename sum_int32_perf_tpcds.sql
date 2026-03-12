-- =============================================================================
-- SUM(int32) SVE 测试 - 使用 TPC-DS 数据集
-- =============================================================================
-- 执行: spark-sql --master yarn -f sum_int32_perf_tpcds.sql --database tpcds_bin_partitioned_varchar_parquat_1000
--
-- 使用 REPARTITION 阻断 pushdown，使聚合走 hashAggUpdateSVEWithCharForNormalInt32
-- =============================================================================

USE tpcds_bin_partitioned_varchar_parquat_1000;

SELECT /*+ REPARTITION(200) */
    ss_store_sk,
    SUM(ss_quantity) AS total_quantity
FROM store_sales
GROUP BY ss_store_sk;
