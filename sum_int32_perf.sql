-- Performance test for SUM(int32) with SVE optimization
-- Database: tpcds_bin_partitioned_varchar_parquat_1000 (TPC-DS 1000GB)
-- Run: spark-sql --master yarn -f sum_int32_perf.sql --database tpcds_bin_partitioned_varchar_parquat_1000
--
-- This query triggers the SVE-optimized int32 SUM path in hash aggregation:
-- - store_sales.ss_quantity is INTEGER (int32)
-- - GROUP BY creates multiple groups, using hash aggregation
-- - SUM(ss_quantity) uses the SVE-optimized sum int32 implementation
--
-- For stable performance results, run this query 3-5 times and take the median.

USE tpcds_bin_partitioned_varchar_parquat_1000;

-- Hash aggregation with SUM(int32): processes full store_sales, sums ss_quantity per store
SELECT
    ss_store_sk,
    SUM(ss_quantity) AS total_quantity
FROM store_sales
GROUP BY ss_store_sk;
