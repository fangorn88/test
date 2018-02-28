""" This modules takes the feature space as input and gives the missions as output """
#########################################################################
# ---------------------------------------Importing Various libraries-------
#########################################################################
import ast
import configparser
import logging
import pyspark.sql.functions as F
import sys
from numpy import array
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import concat
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.window import Window


# Check for environment
job_file = [
    x for x in sys.argv if x.endswith('py') and not x.endswith('__main__.py')]
environment = True if len(job_file) > 0 else False


# Setup if the job is run using spark-submit
if environment is True:
    # Create the SparkContext object and the SQLContext or HiveContext
    sc = SparkContext()
    sqlContext = HiveContext(sc)

# Reading config file
app_config = configparser.ConfigParser()
app_config.read(u'global_mission.ini')

log_lvl = str(app_config["input"]["log_lvl"])
logging.basicConfig(level=logging.NOTSET, datefmt='%I:%M:%S')
logging.getLogger('dunnhumby').setLevel(eval("logging." + log_lvl))

def get_zeroes(c):
    return c.zfill(16)

zero = F.udf(get_zeroes,StringType())



def get_compress(c):
    return c.replace(" ","")
compress = F.udf(get_compress,StringType())



    
sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
sqlContext.sql("SET spark.sql.parquet.compression.codec=snappy")

sqlContext.sql('use x5_ru_analysis')

''''
df = sqlContext.read.parquet("agg_all_x5_business_vsmoking")

df.registerTempTable("df")

week_list = [201636,	201637,	201638,	201639,	201640,	201641,	201642,	201643,	201644,	201645,	201646,	201647,	201648,	201649,	201650,	201651,	201652,	201653,	201701,	201702,	201703,	201704,	201705,	201706,	201707,	201708,	201709,	201710,	201711,	201712,	201713,	201714,	201715,	201716,	201717,	201718,	201719,	201720,	201721,	201722,	201723,	201724,	201725,	201726,	201727,	201728,	201729,	201730,	201731,	201732,	201733,	201734,	201735]

for i in week_list:
    if i == 201636:
        x5_split_master_i = sqlContext.read.parquet("x5_split_master_{}".format(i))
        df = x5_split_master_i
    else:
        x5_split_master_i = sqlContext.read.parquet("x5_split_master_{}".format(i))
        df = df.unionAll(x5_split_master_i)    

df.registerTempTable("df")

trans_master = sqlContext.sql("""
SELECT DISTINCT A.BASKET_ID,A.ITEM_SPEND,BANNER_CODE,ITEM_QTY
FROM x5_ru_analysis.X5_trans_master A
INNER JOIN
  (SELECT *
   FROM x5_ru_inbound.x5_store_c
   WHERE banner_code IN ('D',
                         'S',
                         'H')) B ON a.store_code = b.store_code
WHERE week_id BETWEEN 201636 AND 201735
""")

#201710 AND 201735   201723 AND 201735

trans_master.registerTempTable("trans_master")

rule_mapping = sqlContext.createDataFrame(pd.read_excel("macro_micro_lookup_split.xlsx"))
rule_mapping.write.saveAsTable("ak_macro_x5_lookup_split",
                                        mode='overwrite')


week_list = [1,2,3,4,5,6]

for i in week_list:
    if i == 1:
        x5_split_master_i = sqlContext.read.parquet("agg_all_x5_13_{}".format(i))
        df = x5_split_master_i
    else:
        x5_split_master_i = sqlContext.read.parquet("agg_all_x5_13_{}".format(i))
        df = df.unionAll(x5_split_master_i)    

df.registerTempTable("df")

rule_mapping = sqlContext.createDataFrame(pd.read_excel("macro_micro_lookup_new.xlsx"))
sqlContext.sql("""drop table if exists ak_macro_x5_lookup_new""")
rule_mapping.write.saveAsTable("ak_macro_x5_lookup_new",
                                        mode='overwrite')

week_list = [1,2,3,4,5,6,7]

for i in week_list:
    if i == 1:
        x5_split_master_i = sqlContext.read.parquet("new_miss_tag_{}".format(i))
        df = x5_split_master_i
    else:
        x5_split_master_i = sqlContext.read.parquet("new_miss_tag_{}".format(i))
        df = df.unionAll(x5_split_master_i)    

df.registerTempTable("trans_all")

'''

#sqlContext.sql("""drop table if exists AK_ALL_BASK_""")
df = sqlContext.sql("""
SELECT MACRO_MISSION,
       BANNER_CODE,
       HOUR_OF_DAY,
       DAY_OF_WEEK,
       SUM(ITEM_SPEND) AS SPEND,
       COUNT(*) AS BASKETS
FROM AK_ALL_BASK_TAG_1636_1748_1  A
INNER JOIN
  (SELECT  BASKET_ID,
                   STORE_CODE,
                   HOUR_OF_DAY,
                   DAY_OF_WEEK   
   FROM x5_ru_analysis.X5_trans_master
   WHERE week_id BETWEEN 201650 AND 201748
   GROUP BY BASKET_ID,
            STORE_CODE,
            HOUR_OF_DAY,
            DAY_OF_WEEK) B ON A.TRANSACTION_CODE = B.BASKET_ID
WHERE WEEK_ID BETWEEN 201650 AND 201748
GROUP BY MACRO_MISSION,
         BANNER_CODE,
         HOUR_OF_DAY,
         DAY_OF_WEEK
""")  

df.toPandas().to_excel('new_miss_tag_KPIs_day_time.xlsx')

if environment is True:
    sc.stop()
