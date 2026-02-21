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
    .appName("CaipanDataQuery") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .getOrCreate()

# 2. 定义txt输出文件路径（保存到挂载目录，宿主机可直接查看）
output_txt_path = "/opt/project/wenshu2.txt"

# 3. 启动输出重定向（所有print/show/printSchema都会写入txt）
tee = TeeOutput(output_txt_path)
sys.stdout = tee

# 4. 定义所有ORC文件路径
source_path = "/opt/lol_data/*裁判文书数据*.orc"
# 5. 批量读取所有ORC文件
print("===== 开始读取ORC文件 =====")
orc_dfs = spark.read.orc(source_path)

# 6. 显示所有文件的基本信息（总行数）
keyword = "%囚禁%"
res = orc_dfs.select("案件名称", "所属地区", "全文") \
        .filter(col("全文").like(keyword))
print(f"\n===== 开始执行检索: {keyword} =====")

res.write.mode("overwrite") \
   .option("header", "true") \
   .option("sep", "\t") \
   .csv("/opt/project/output_full_text")
# 9. 结束处理：恢复stdout + 关闭文件
print("\n===== 分析完成，结果已写入：{} =====".format(output_txt_path))
sys.stdout = tee.original_stdout  # 恢复原始输出
tee.close()  # 关闭txt文件
spark.stop()  # 停止SparkSession
