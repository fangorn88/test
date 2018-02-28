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

SELECT CHAIN_NAME,CHAIN_FORMAT,FED_NAME,
       SUM(SPEND) AS SPEND,
       COUNT(*) AS BASKETS,
       COUNT(DISTINCT B.PERSON_ID) AS CUSTOMERS
FROM DF1 A
INNER JOIN
  (SELECT RECEIPT_ID,CHAIN_FORMAT,FED_NAME,HOUSEHOLD_ID,PERSON_ID,CHAIN_NAME,
          SUM(SALES) AS SPEND
   FROM FINAL_MISSIONS A
   INNER JOIN AK_ROMIR_CAT_NS B ON A.CATEGORY_NAME = B.CATEGORY_NAME
   WHERE WEEK_ID > 201701
   GROUP BY RECEIPT_ID,CHAIN_NAME,CHAIN_FORMAT,FED_NAME,HOUSEHOLD_ID,PERSON_ID) B ON A.TRANSACTION_CODE = B.RECEIPT_ID
INNER JOIN AK_ROMIR_USER U ON B.PERSON_ID = U.PERSON_ID
LEFT JOIN AK_ROMIR_HHD_LOW H ON B.HOUSEHOLD_ID = H.HH_ID
LEFT JOIN AK_MACRO_X5_LOOKUP_NEW D ON A.SM_CONCAT = D.SM_CONCAT
WHERE H.HH_ID IS NULL 
GROUP BY CHAIN_NAME,CHAIN_FORMAT,FED_NAME

#201710 AND 201735   201723 AND 201735

trans_master.registerTempTable("trans_master")

rule_mapping = sqlContext.createDataFrame(pd.read_excel("macro_micro_lookup_split.xlsx"))
rule_mapping.write.saveAsTable("ak_macro_x5_lookup_split",
                                        mode='overwrite')


sample = sqlContext.read.parquet("agg_all_x5_business_v2")

sample.registerTempTable("sample")

sqlContext.sql("""drop table if exists ak_romir_user""")
b = sqlContext.createDataFrame(pd.read_csv("dun_user_status_updated.csv"))
b.write.saveAsTable("ak_romir_user",
                                        mode='overwrite')

sqlContext.sql("""drop table if exists ak_romir_hhd_low""")
need_state = sqlContext.createDataFrame(pd.read_csv("HH_with_low_quality.csv"))
need_state.write.saveAsTable("ak_romir_hhd_low",
                                        mode='overwrite')

df1 = sqlContext.read.parquet("agg_all_romir_ak_n")

df1.registerTempTable("df1")

sqlContext.sql("""drop table if exists ak_romir_cat_ns""")
need_state = sqlContext.createDataFrame(pd.read_excel("romir_ns_map.xlsx"))
need_state.write.saveAsTable("ak_romir_cat_ns",
                                        mode='overwrite')


'''


#sqlContext.sql("""drop table if exists AK_ALL_BASK_""")
df = sqlContext.sql("""
SELECT MNTH,DISTRICT,
       A.BANNER_CODE,
       MACRO_MISSION,
       SUM(ITEM_SPEND) AS SPEND,
       COUNT(DISTINCT TRANSACTION_CODE) AS BASKETS,
       COUNT(DISTINCT A.CUSTOMER_ID) AS CUSTOMERS
FROM
  (SELECT SM_CONCAT,
          TRANSACTION_CODE,DISTRICT,
          MACRO_MISSION,
          BANNER_CODE,
          CUSTOMER_ID,
          ITEM_SPEND,
          STORE_CODE,
          CONCAT(YEAR(SHOP_DATE),MONTH(SHOP_DATE)) AS MNTH
   FROM AK_ALL_BASK_TAG_1636_1748_1
   WHERE (SHOP_DATE BETWEEN '2017-09-01' AND '2017-11-30')
     AND CUSTOMER_ID != ""
     AND CUSTOMER_ID IS NOT NULL) A
INNER JOIN (SELECT CUSTOMER_ID,BANNER_CODE FROM AK_CUST_EXCL_1 WHERE PERC BETWEEN 0.01001 AND 0.99) D 
ON A.CUSTOMER_ID = D.CUSTOMER_ID AND A.BANNER_CODE = D.BANNER_CODE 
GROUP BY MNTH,DISTRICT,
         A.BANNER_CODE,
         MACRO_MISSION
GROUPING
SETS ((MNTH,DISTRICT,
       A.BANNER_CODE,
       MACRO_MISSION),(MNTH,
                       DISTRICT))
""")  

df.toPandas().to_excel('new_miss_tag_KPIs_cust_region.xlsx')

if environment is True:
    sc.stop()
