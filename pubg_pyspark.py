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
    .appName("PUBG_Analysis") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .getOrCreate()

# 2. 定义txt输出文件路径（保存到挂载目录，宿主机可直接查看）
output_txt_path = "/opt/project/pubg_result2.txt"

# 3. 启动输出重定向（所有print/show/printSchema都会写入txt）
tee = TeeOutput(output_txt_path)
sys.stdout = tee

# 4. 定义所有ORC文件路径
orc_files = {
    "stats0": "/opt/lol_data/agg_match_stats_0.orc",
    "kill0": "/opt/lol_data/kill_match_stats_final_0.orc"
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
stats0 = orc_dfs['stats0']
stats0.createOrReplaceTempView("stats0")
kill0 = orc_dfs['kill0']
kill0.createOrReplaceTempView("kill0")
print("\n--- 分析主题 1: 决赛对枪胜负关系---")
first_id_df = stats0.select("match_id","player_name").filter(col("team_placement") == 1)
first_id_df.createOrReplaceTempView("1st_id")
second_id_df = stats0.select("match_id","player_name").filter(col("team_placement") == 2)
second_id_df.createOrReplaceTempView("2nd_id")
spark.sql("""
WITH win_weapon AS (
    -- 统计吃鸡玩家使用的武器
    SELECT 
        k.killed_by AS weapon
        ,w.match_id
        ,k.victim_name
    FROM kill0 k
    INNER JOIN 1st_id w ON k.match_id = w.match_id AND k.killer_name = w.player_name
    WHERE k.killed_by NOT IN ('Bluezone', 'Down and Out', 'Falling', 'Hit by Car', 'Pickup Truck', 'death.WeapSawnoff_C', 'Buggy', 'death.ProjMolotov_DamageField_C', 'Boat', 'Sickle', 'death.ProjMolotov_C', 'death.Buff_FireDOT_C', 'Motorbike', 'Crowbar') and k.victim_placement = 2
),
2nd_weapon AS (
    Select weapon
        ,match_id
        ,killer_name
    from 
    (
    -- 统计第二名玩家使用的武器
    SELECT 
        k.killed_by AS weapon
        ,n.match_id
        ,k.killer_name
        ,row_number() over(partition by n.match_id order by k.victim_placement) as rk
    FROM 2nd_id n
    left JOIN kill0 k ON k.match_id = n.match_id AND k.killer_name = n.player_name
    WHERE k.killed_by NOT IN ('Bluezone', 'Down and Out', 'Falling', 'Hit by Car', 'Pickup Truck', 'death.WeapSawnoff_C', 'Buggy', 'death.ProjMolotov_DamageField_C', 'Boat', 'Sickle', 'death.ProjMolotov_C', 'death.Buff_FireDOT_C', 'Motorbike', 'Crowbar')
) t
    where rk = 1)
SELECT 
    w.weapon as kill_weapon
    ,n.weapon as 2nd_weapon
    ,count(w.match_id) as match_cnt
    ,sum(count(w.match_id)) over(partition by w.weapon) as total_match_cnt
    ,round(count(w.match_id) / sum(count(w.match_id)) over(partition by w.weapon),4) as pct
FROM win_weapon w
inner join 2nd_weapon n
    on case when w.match_id is null then concat('test_', rand()) else w.match_id end = n.match_id and w.victim_name = n.killer_name
group by w.weapon,n.weapon
order by total_match_cnt desc, match_cnt desc
""").show(truncate=False,n=1000)




# 2. 用户流失分析
# ==============================================================================
print("\n--- 分析主题 2: 用户流失分析 ---")
#注册三个月流失分析
"""
with a as()
    
"""
# 9. 结束处理：恢复stdout + 关闭文件
print("\n===== 分析完成，结果已写入：{} =====".format(output_txt_path))
sys.stdout = tee.original_stdout  # 恢复原始输出
tee.close()  # 关闭txt文件
spark.stop()  # 停止SparkSession
