# -*- coding: utf-8 -*-
# 导入必要依赖（新增sys模块用于输出重定向）
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, round, from_unixtime, to_timestamp,  regexp_replace, lit, count, sum,countDistinct,regexp_extract, lower, trim
from tabulate import tabulate

# 自定义输出重定向类：同时输出到控制台 + 写入txt文件
class TeeOutput:
    def __init__(self, file_path):
        # 保存原始stdout
        self.original_stdout = sys.stdout
        # 打开txt文件（utf-8编码，覆盖写入，Linux兼容）
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        # 同时写入文件和控制台
        self.original_stdout.write(message)
        self.file.write(message)
        # 立即刷新，避免输出缓存
        self.file.flush()
    
    def flush(self):
        # 兼容Spark的输出刷新
        self.original_stdout.flush()
        self.file.flush()
    
    def close(self):
        # 恢复原始stdout + 关闭文件
        sys.stdout = self.original_stdout
        self.file.close()

# 1. 创建SparkSession（单机模式）
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("tft_Analysis") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .getOrCreate()

# 2. 定义txt输出文件路径（保存到挂载目录，宿主机可直接查看）
output_txt_path = "/opt/project/tft_result2.txt"

# 3. 启动输出重定向（所有print/show/printSchema都会写入txt）
tee = TeeOutput(output_txt_path)
sys.stdout = tee

# 4. 定义所有ORC文件路径
orc_files = {
    "match": "/opt/lol_data/TFT_MatchData.orc",
}

# 5. 批量读取所有ORC文件
print("===== 开始读取ORC文件 =====")
orc_dfs = {}
for file_name, file_path in orc_files.items():
    try:
        orc_dfs[file_name] = spark.read.orc(file_path)
        print("成功读取文件：{}".format(file_name))
    except Exception as e:
        error_msg = "读取文件{}失败：{}".format(file_name, str(e))
        print(error_msg)

# 6. 显示所有文件的基本信息（总行数）
print("\n=== 1. 所有ORC文件数据基本信息 ===")
for file_name, df in orc_dfs.items():
    row_count = df.count()
    print("{} - 总行数：{}".format(file_name, row_count))

# 7. 显示每个文件的字段结构+5行头数据+5行尾数据
print("\n=== 2. 所有ORC文件字段结构与数据预览 ===")
for file_name, df in orc_dfs.items():
    print("\n" + "-"*50)
    print("文件名称：{}".format(file_name))
    print("-"*50)
    
    # 7.1 字段结构（列名+属性）
    print("【字段结构（列名+数据类型）】")
    df.printSchema()
    
    # 7.2 前5行数据
    print("\n【前5行数据】")
    df.show(5, truncate=True)
    
    # 7.3 后5行数据
    print("\n【后5行数据】")
    tail_rows = df.tail(5)
    if tail_rows:
        spark.createDataFrame(tail_rows, schema=df.schema).show(5, truncate=True)
    else:
        print("该文件无数据")
match = orc_dfs["match"]
match.createOrReplaceTempView("match")
print("\n=== 3. 英雄与单装备收益 ===")
df_god_items = spark.sql("""--sql
-- WITH exploded_champs AS (
--     SELECT 
--         gameId
--         ,tier
--         ,Ranked
--         -- 解析嵌套JSON：英雄名为Key，详情为Value
--         ,explode(from_json(champion,'MAP<STRING,STRUCT<items:ARRAY<INT>,star:INT>>')) as (champ_name, details)   --MAP需要加<>，STRUCT需要加<>ARRAY需要加<>
--     FROM match
--     ),
--     champ_item_flat AS (
--         SELECT 
--             tier
--             ,Ranked
--             ,champ_name
--             ,details.star as star_level
--             -- 炸开装备数组
--             ,explode_outer(details.items) as item_id
--         FROM exploded_champs
--     ),
    with champ_item_flat as(
        SELECT 
            tier
            ,Ranked
            ,champ_name
            ,details.star as star_level
            ,item_id
            ,size(details.items) as item_count
        FROM match
        LATERAL VIEW explode(from_json(champion,'MAP<STRING,STRUCT<items:ARRAY<INT>,star:INT>>')) t1 as champ_name, details   --MAP需要加<>，STRUCT需要加<>ARRAY需要加<>
        LATERAL VIEW explode_outer(details.items) t2 as item_id
    ),
    zero_item AS (
    -- 计算0装备基础胜率
    SELECT 
        tier
        ,champ_name
        ,AVG(IF(Ranked = 1, 1.0, 0.0)) as zero_item_win_rate
        ,count(*) as zero_count
    FROM champ_item_flat
    WHERE item_count = 0
    GROUP BY tier, champ_name
    ),
    one_item AS (
    -- 计算单装备基础胜率
    SELECT 
        tier
        ,champ_name
        ,item_id
        ,AVG(IF(Ranked = 1, 1.0, 0.0)) as one_item_win_rate
        ,count(*) as one_count
    FROM champ_item_flat
    WHERE item_count = 1
    GROUP BY tier, champ_name, item_id
    )
    select
        zero_item.tier
        ,zero_item.champ_name
        ,one_item.item_id
        ,round(zero_item_win_rate, 4) as zero_item_win_rate
        ,zero_count
        ,round(one_item_win_rate, 4) as one_item_win_rate
        ,one_count
        ,round(one_item_win_rate - zero_item_win_rate, 4) as lift_score
    from zero_item
    inner join one_item
    on zero_item.tier = one_item.tier and zero_item.champ_name = one_item.champ_name
    where one_count > 50
    order by lift_score desc
""")
df_god_items.show(n=10)
pandas_df = df_god_items.toPandas()
local_csv_path = "/opt/project/tft_result.csv"
pandas_df.to_csv(local_csv_path, index=False, encoding='utf-8-sig')
print("\n=== 4. 英雄与多装备收益 ===")
df_items = spark.sql("""--sql
    with base_info AS (
    SELECT 
        tier
        ,Ranked
        ,champ_name
        ,size(details.items) as item_count
        -- 核心：对装备数组排序并转为字符串，确保 [1,2] 和 [2,1] 视为同一套装
        ,CASE 
            WHEN size(details.items) = 0 THEN 'NONE'
            ELSE array_join(array_sort(details.items), '-')
        END as item_set
    FROM match
    LATERAL VIEW explode(from_json(champion, 'MAP<STRING,STRUCT<items:ARRAY<INT>,star:INT>>')) t1 as champ_name, details
    ),
    combo_metrics AS (
    -- 第二步：计算每种【英雄+装备数+具体套装】的胜率指标
    SELECT 
        tier
        ,champ_name
        ,item_count
        ,item_set
        ,COUNT(*) as combo_count
        ,AVG(IF(Ranked = 1, 1.0, 0.0)) as win_rate
        ,AVG(IF(Ranked <= 4, 1.0, 0.0)) as top4_rate
    FROM base_info
    GROUP BY tier, champ_name, item_count, item_set
    ),
    com_rnk as(
        select *
        ,row_number() over(partition by tier, champ_name, item_count order by item_count desc) as rnk
        from combo_metrics
        where combo_count > 50
    ),
    filter_rnk as(
        select *
        from com_rnk
        where  (item_count = 0 AND rnk = 1) OR
        (item_count = 1 AND rnk <= 3) OR
        (item_count = 2 AND rnk <= 5) OR
        (item_count = 3 AND rnk <= 5)
    ),
    naked_rnk as(
        select *
        from com_rnk
        where  (item_count = 0 AND rnk = 1)
    )
    select 
    f.tier
    ,f.champ_name
    ,f.item_count
    ,f.item_set
    ,f.combo_count
    ,f.rnk
    ,round(f.win_rate, 4) as current_win_rate
    -- 关联计算：找到同段位同英雄 0 装备时的胜率作为基准
    ,round(b.win_rate, 4) as base_naked_wr
    ,round(f.win_rate - b.win_rate, 4) as total_lift
    FROM filter_rnk f
    LEFT JOIN naked_rnk b ON f.tier = b.tier 
        AND f.champ_name = b.champ_name
    ORDER BY f.tier, f.champ_name, f.item_count ASC, f.win_rate DESC;
""")
pandas_df = df_items.toPandas()
local_csv_path = "/opt/project/tft_result2.csv"
pandas_df.to_csv(local_csv_path, index=False, encoding='utf-8-sig')
# 9. 结束处理：恢复stdout + 关闭文件
print("\n===== 分析完成，结果已写入：{} =====".format(output_txt_path))
sys.stdout = tee.original_stdout  # 恢复原始输出
tee.close()  # 关闭txt文件
spark.stop()  # 停止SparkSession
