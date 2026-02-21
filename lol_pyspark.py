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
    .appName("LOL_ORC_Analysis") \
    .getOrCreate()

# 2. 定义txt输出文件路径（保存到挂载目录，宿主机可直接查看）
output_txt_path = "/opt/project/lol_result2.txt"

# 3. 启动输出重定向（所有print/show/printSchema都会写入txt）
tee = TeeOutput(output_txt_path)
sys.stdout = tee

# 4. 定义所有ORC文件路径
orc_files = {
    "champs": "/opt/lol_data/champs.orc",
    "matches": "/opt/lol_data/matches.orc",
    "participants": "/opt/lol_data/participants.orc",
    "stats1": "/opt/lol_data/stats1.orc",
    "stats2": "/opt/lol_data/stats2.orc",
    "teambans": "/opt/lol_data/teambans.orc",
    "teamstats": "/opt/lol_data/teamstats.orc"
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

# 8. stats1.orc KDA核心分析
if "stats1" in orc_dfs:
    stats1_df = orc_dfs["stats1"]
    # 计算KDA
    stats1_df = stats1_df.withColumn(
        "kda",
        when(col("deaths") == 0, col("kills") + col("assists"))
        .otherwise(round((col("kills") + col("assists")) / col("deaths"), 2))
    )

    print("\n=== 3. stats1.orc KDA分析 ===")
    # KDA前10数据
    print("\n【KDA前10数据】")
    stats1_df.select("id", "win", "kills", "deaths", "assists", "kda") \
        .orderBy(col("kda").desc()) \
        .show(10, truncate=True)
    
    # 胜负方平均KDA对比
    # print("\n【胜负方平均KDA对比】")
    # stats1_df.groupBy("win") \
    #     .agg(
    #         round(col("kda").avg(), 2).alias("avg_kda"),
    #         round(col("kills").avg(), 1).alias("avg_kills")
    #     ) \
    #     .show(truncate=True)
if "matches" in orc_dfs:
    match_df = orc_dfs["matches"]
    print("=== 最早的 5 行 ===")
    match_df.select('*',from_unixtime(col("creation").cast("long") / 1000).alias("creation_date")).orderBy(col("creation_date")).limit(5).show(truncate=True)
    print("=== 最早的 5 行 ===")
    match_df.select('*',from_unixtime(col("creation").cast("long") / 1000).alias("creation_date")).orderBy(col("creation_date").desc()).limit(5).show(truncate=True)
    
if True:
    # 7.6版本加里奥重做，分析效果如何
    champs_df = orc_dfs["champs"]
    match_df = orc_dfs["matches"]
    participants_df = orc_dfs["participants"]
    stats1_df = orc_dfs["stats1"]
    stats2_df = orc_dfs["stats2"]  
    teambans_df = orc_dfs["teambans"]  
    teamstats_df = orc_dfs["teamstats"]
    stats_df = stats1_df.unionByName(stats2_df)
    
    participants_df.createOrReplaceTempView("participants")
    champs_df.createOrReplaceTempView("champs")
    teambans_df.createOrReplaceTempView("teambans")
    stats_df.createOrReplaceTempView("stats")
    match_filter = match_df.select('id','version',regexp_extract(col('version'), '^7\\.[567]', 0).alias('version_adj')).filter(col('version').rlike('^7\\.[567]')
    )
    match_filter.createOrReplaceTempView("match_filter")
    result_sql = spark.sql("""
    WITH 
    participants_with_champs AS (
        SELECT 
            p.*,
            c_pick.name AS pick_name
        FROM participants p
        LEFT JOIN champs c_pick 
            ON p.championid = c_pick.id
    ),
    teambans_with_champs AS (
        SELECT 
            b.*,
            c_ban.name AS ban_name
        FROM teambans b
        LEFT JOIN champs c_ban 
            ON b.championid = c_ban.id
    ),
    participants_with_stats AS (
        SELECT 
            p.*,
            s.win
        FROM participants_with_champs p
        LEFT JOIN stats s 
            ON p.id = s.id
    ),
    play_stats AS (
        SELECT 
            m.version_adj,
            COUNT(DISTINCT m.id) AS total_matches,
            COUNT(DISTINCT CASE WHEN LOWER(TRIM(p.pick_name)) = 'galio' THEN p.matchid END) AS play_count,
            COUNT(DISTINCT CASE WHEN LOWER(TRIM(p.pick_name)) = 'galio' AND p.win = 1 THEN p.matchid END) AS win_count
        FROM match_filter m
        LEFT JOIN participants_with_stats p 
            ON m.id = p.matchid
        GROUP BY m.version_adj
    ),
    ban_stats AS (
        SELECT 
            m.version_adj,
            COUNT(DISTINCT CASE WHEN LOWER(TRIM(b.ban_name)) = 'galio' THEN b.matchid END) AS ban_count
        FROM match_filter m
        LEFT JOIN teambans_with_champs b 
            ON m.id = b.matchid
        GROUP BY m.version_adj
    )
    SELECT 
        p.version_adj,
        p.total_matches,
        p.play_count,
        b.ban_count,
        p.win_count,
        p.play_count / p.total_matches AS play_rate,
        b.ban_count / p.total_matches AS ban_rate,
        p.win_count / p.play_count AS win_rate
    FROM play_stats p
    JOIN ban_stats b 
        ON p.version_adj = b.version_adj
    ORDER BY p.version_adj;
    """)

    result_sql.show()


# 9. 结束处理：恢复stdout + 关闭文件
print("\n===== 分析完成，结果已写入：{} =====".format(output_txt_path))
sys.stdout = tee.original_stdout  # 恢复原始输出
tee.close()  # 关闭txt文件
spark.stop()  # 停止SparkSession
