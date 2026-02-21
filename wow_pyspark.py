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
output_txt_path = "/opt/project/wow_result2.txt"

# 3. 启动输出重定向（所有print/show/printSchema都会写入txt）
tee = TeeOutput(output_txt_path)
sys.stdout = tee

# 4. 定义所有ORC文件路径
orc_files = {
    "locatc": "/opt/lol_data/wow_location_coords.orc",
    "locat": "/opt/lol_data/wow_locations.orc",
    "wowah": "/opt/lol_data/wow_wowah_data.orc",
    "zone": "/opt/lol_data/wow_zones.orc"
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
        
print("\n=== 3. 满级团本职业占比分析职业平衡 ===")
locatc = orc_dfs["locatc"]
locat = orc_dfs["locat"]
wowah = orc_dfs["wowah"]
zone = orc_dfs["zone"]
locatc.createOrReplaceTempView("locatc")
locat.createOrReplaceTempView("locat")
wowah.createOrReplaceTempView("wowah")
zone.createOrReplaceTempView("zone")
lv80_players = wowah.select("char_id", "charclass","level", "zone").filter(col("level") == 80)
lv80_players.createOrReplaceTempView("lv80_players")
total_players = wowah.select("char_id", "charclass","level")
total_players.createOrReplaceTempView("total_players")
dung_zone = zone.select("Zone", "Type").filter((col("Type") == "Dungeon") & (col("Size").isin('10', '20', '25', '40')))
dung_zone.createOrReplaceTempView("dung_zone")

spark.sql(
    """ --sql
        with lv80_dung as (
            select charclass
            ,count(charclass) as dung_count
            ,round(count(charclass) / sum(count(charclass)) over(),4) as dung_ratio
            from 
            (
            select
                p.charclass,
                p.char_id
            from dung_zone z
            inner join lv80_players p on p.zone = z.Zone
            group by charclass, char_id
            ) t
            group by charclass
        ),
        lv80_total as (
            select charclass
            ,count(if(level == 80, charclass, null)) as lv80_count
            ,round(count(if(level == 80, charclass, null)) / sum(count(if(level == 80, charclass, null))) over(),4) as lv80_ratio
            from
                (
                select
                    charclass,
                    char_id,
                    level
                from total_players
                group by charclass, char_id, level
                ) t
            group by charclass
        )
        select lv80_dung.charclass
            ,lv80_dung.dung_count
            ,lv80_dung.dung_ratio
            ,lv80_total.lv80_count
            ,lv80_total.lv80_ratio
            ,round(lv80_dung.dung_ratio / lv80_total.lv80_ratio,4) as dung_ratio_ratio
        from lv80_dung
        left join lv80_total on lv80_dung.charclass = lv80_total.charclass
        order by dung_ratio_ratio desc
    """
).show(n=1000)

print("\n=== 4. 有无公会对角色练度关系（加入公会是否能快速练级减少流失） ===")
spark.sql(
    """--sql
        with char_lv as(
            select char_id
            ,if(guild = -1, '无公会', '有公会') as guild_flag
            ,case
                when level = 80 then '80'
                when level >= 68 then '68-79'  --诺森德巫妖王
                when level >= 58 then '58-67'  --外域
                when level >= 30 then '30-57'  --艾泽拉斯
                when level >= 11 then '11-29'  --无坐骑练级
                else '01-10' end as lv_bin
            from(
                select
                    char_id
                    ,level
                    ,guild
                    ,row_number() over(partition by char_id order by timestamp desc) as row
                from wowah
            )
            where row = 1
        )
        select lv_bin
            ,guild_flag
            ,count(char_id) as char_count
            ,round(count(char_id) / sum(count(char_id)) over(partition by lv_bin),4) as char_ratio
        from char_lv
        group by lv_bin, guild_flag
        order by lv_bin, guild_flag
    """
).show(n=1000)
print("\n=== 5. 200801-200811期间各职业处于不同练度等级的流失比例最大 ===")
spark.sql(
    """--sql
        with char_lv as(
            select char_id
            ,charclass
            ,case
                when level = 80 then '80'
                when level >= 68 then '68-79'  --诺森德巫妖王
                when level >= 58 then '58-67'  --外域
                when level >= 30 then '30-57'  --艾泽拉斯
                when level >= 11 then '11-29'  --无坐骑练级
                else '01-10' end as lv_bin
            ,if((datediff('2008-12-30', timestamp) > 30 and last_timestamp is not null) or (timestamp < '2008-11-30' and last_timestamp is null), 1, 0) as lv_change_flag
            from(
                select
                    char_id
                    ,charclass
                    ,level
                    ,timestamp
                    ,lag(timestamp) over(partition by char_id order by timestamp) as last_timestamp
                    ,row_number() over(partition by char_id order by timestamp desc) as row
                from wowah
            )
            where row = 1
        )
        select charclass
            ,lv_bin
            ,count(char_id) as char_count
            ,sum(lv_change_flag) as lv_change_count
            ,round(sum(lv_change_flag) / count(char_id),4) as lv_change_ratio
        from char_lv
        group by charclass, lv_bin
        order by charclass, lv_bin
    """
).show(n=1000)
# 9. 结束处理：恢复stdout + 关闭文件
print("\n===== 分析完成，结果已写入：{} =====".format(output_txt_path))
sys.stdout = tee.original_stdout  # 恢复原始输出
tee.close()  # 关闭txt文件
spark.stop()  # 停止SparkSession
