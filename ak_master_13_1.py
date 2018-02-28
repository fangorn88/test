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
from pyspark.sql.functions import udf


# Check for environment
job_file = [
    x for x in sys.argv if x.endswith('py') and not x.endswith('__main__.py')]
environment = True if len(job_file) > 0 else False


# Setup if the job is run using spark-submit
if environment is True:
    # Create the SparkContext object and the SQLContext or HiveContext
    sc = SparkContext()
    sqlContext = HiveContext(sc)
'''
# Reading config file
app_config = configparser.ConfigParser()
app_config.read(u'micro_mission.ini')


v_small = str(app_config["rules"]["very_small"]).split(",")
small = str(app_config["rules"]["small"]).split(",")
medium = str(app_config["rules"]["medium"]).split(",")
large = str(app_config["rules"]["large"]).split(",")
v_large = int(app_config["rules"]["very_large"])
num_ns = int(app_config["rules"]["num_ns"])
'''

def get_zeroes(c):
    return c.zfill(16)

zero = F.udf(get_zeroes,StringType())



def get_compress(c):
    return c.replace(" ","")
compress = F.udf(get_compress,StringType())



    
sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
sqlContext.sql("SET spark.sql.parquet.compression.codec=snappy")

sqlContext.sql('use x5_ru_analysis')


need_state  = sqlContext.sql("""
SELECT A.*,B.NS_HARM FROM 
(SELECT A.*,CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH_1) AS NEW_CAT 
FROM x5_ru_analysis.ak_prod_ovr_1 A) A INNER JOIN 
X5_RU_ANALYSIS.AK_NS_MAP_OVR_SPLIT B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL 
""")
need_state = need_state.withColumn("need_state",F.upper(compress("NS_HARM")))
need_state = need_state.select("product_code","need_state")
# need_state = need_state.withColumn("format_code",F.upper(compress("format_code")))

'''
rule_mapping = sqlContext.createDataFrame(pd.read_excel("X5_lookup_micro_business_v3.xlsx"))
rule_mapping.write.saveAsTable("ak_micro_ns_x5_lookup_business_3",
                                        mode='overwrite')
'''
rule_mapping = sqlContext.table("ak_micro_ns_x5_lookup_business_3")

#rule_mapping  = sqlContext.table("ak_micro_ns_x5_lookup ")
for i in ["need_state","Micro_qualifier_2","Micro_qualifier_3_5","Micro_qualifier_6_7"]:
    rule_mapping = rule_mapping.withColumn(i,F.upper(compress(i)))



df_final = sqlContext.sql(""" 
SELECT DISTINCT BASKET_ID AS TRANSACTION_CODE,
       NS_HARM AS NEED_STATE,
       ITEM_SPEND AS NET_SPEND_AMT,
       CATEGORY_DESC_ENG AS PROD_HIER_L30_CODE,
       A.PRODUCT_CODE AS CATEGORY_NAME
FROM AK_TRANS_52weeks A
INNER JOIN
  (SELECT A.*,CASE WHEN S.NS_NEW IS NOT NULL THEN NS_NEW ELSE NS_HARM END AS NS_HARM
   FROM
     (SELECT A.*,
             CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH_1) AS NEW_CAT
      FROM x5_ru_analysis.ak_prod_ovr_1 A) A
   INNER JOIN X5_RU_ANALYSIS.AK_NS_MAP_OVR_SPLIT B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL
LEFT JOIN AK_SEMI_MEAT_NS S ON A.SUB_GROUP = S.CODE) B ON A.PRODUCT_CODE = B.PRODUCT_CODE
WHERE item_spend > 0
  AND item_qty > 0
  AND WEEK_ID BETWEEN 201711 AND 201720
""")

#####################################################################################################
# ---------------------------------------------Data Pull-----------------------------------------------
#####################################################################################################
trans_cat_data = df_final

trans_cat_data = trans_cat_data.where(F.col("net_spend_amt")>0)

nss_all = trans_cat_data.join(rule_mapping,"need_state","inner")

nss_all.cache()

nss_all2 = nss_all.groupBy("transaction_code",
                           "need_state","Micro_qualifier_2",
                           "Micro_qualifier_3_5",
                           "Micro_qualifier_6_7")\
                  .agg(F.sum("net_spend_amt").cast(DecimalType(16,4)).alias("ns_spend"))

    
basket_summ = nss_all.groupBy("transaction_code")\
                     .agg(F.sum("net_spend_amt").cast(DecimalType(16,4)).alias("bask_spend"),
                         F.countDistinct("need_state").alias("cnt_needstate"))

nss_all_2 = nss_all2.join(basket_summ,"transaction_code","inner")\
                    .withColumn("perc_ns_spend" , F.col("ns_spend")/F.col("bask_spend") )
nss_all_2.cache()
micro_2 = nss_all_2.groupBy("transaction_code","Micro_qualifier_2")\
                   .agg(F.sum("perc_ns_spend").cast(DecimalType(16,4)).alias("dom_perc_spend_2"))\
                   .withColumnRenamed("Micro_qualifier_2","dom_micro_2")
micro_3_5 = nss_all_2.groupBy("transaction_code","Micro_qualifier_3_5")\
                   .agg(F.sum("perc_ns_spend").cast(DecimalType(16,4)).alias("dom_perc_spend_3_5"))\
                   .withColumnRenamed("Micro_qualifier_3_5","dom_micro_3_5")
micro_6_7 = nss_all_2.groupBy("transaction_code","Micro_qualifier_6_7")\
                   .agg(F.sum("perc_ns_spend").cast(DecimalType(16,4)).alias("dom_perc_spend_6_7"))\
                   .withColumnRenamed("Micro_qualifier_6_7","dom_micro_6_7")
        
# TIE SOLVER FOR MISSIONS AND NEEDSTATES
tie_2 = nss_all.groupBy("Micro_qualifier_2")\
               .agg(F.sum("net_spend_amt").cast(DecimalType(16,4)).alias("spend"))\
               .withColumnRenamed("Micro_qualifier_2","mq")
tie_3_5 = nss_all.groupBy("Micro_qualifier_3_5")\
               .agg(F.sum("net_spend_amt").cast(DecimalType(16,4)).alias("spend"))\
               .withColumnRenamed("Micro_qualifier_3_5","mq")
tie_6_7 = nss_all.groupBy("Micro_qualifier_6_7")\
               .agg(F.sum("net_spend_amt").cast(DecimalType(16,4)).alias("spend"))\
               .withColumnRenamed("Micro_qualifier_6_7","mq")

tie = tie_2.unionAll(tie_3_5).unionAll(tie_6_7).orderBy(F.col("spend").desc())
tie = tie.withColumn("rank",F.row_number().over(Window.partitionBy("mq").orderBy(F.col("spend").desc())))
tie = tie.where(F.col("rank")==1).drop("rank")
tie_solver = tie.withColumn("tie_var",F.rank().over(Window.orderBy("spend"))).drop("spend")


micro_2  = micro_2.join(tie_solver,micro_2.dom_micro_2==tie_solver.mq,"inner").drop("mq")
micro_3_5 = micro_3_5.join(tie_solver,micro_3_5.dom_micro_3_5==tie_solver.mq,"inner").drop("mq")
micro_6_7 = micro_6_7.join(tie_solver,micro_6_7.dom_micro_6_7==tie_solver.mq,"inner").drop("mq")

micro_2_re = micro_2.withColumn("rank",
                                F.rank().over(Window.partitionBy("transaction_code")\
                                .orderBy(F.desc("dom_perc_spend_2"),
                                         "tie_var")))

micro_2_fir = micro_2_re.where(F.col('rank')==1).drop("rank").select(micro_2_re.columns[:4])
micro_2_sec = micro_2_re.where(F.col('rank')==2).drop("rank").select(micro_2_re.columns[:4])

micro_3_5_re = micro_3_5.withColumn("rank",
                                F.rank().over(Window.partitionBy("transaction_code")\
                                .orderBy(F.desc("dom_perc_spend_3_5"),
                                         "tie_var")))
micro_3_5_fir = micro_3_5_re.where(F.col('rank')==1).drop("rank").select(micro_3_5_re.columns[:3])
micro_3_5_sec = micro_3_5_re.where(F.col('rank')==2).drop("rank").select(micro_3_5_re.columns[:3])


micro_6_7_re = micro_6_7.withColumn("rank",
                                F.rank().over(Window.partitionBy("transaction_code")\
                                .orderBy(F.desc("dom_perc_spend_6_7"),
                                         "tie_var")))
micro_6_7_fir = micro_6_7_re.where(F.col('rank')==1).drop("rank").select(micro_6_7_re.columns[:3])
micro_6_7_sec = micro_6_7_re.where(F.col('rank')==2).drop("rank").select(micro_6_7_re.columns[:3])



aggregate_all_fir = micro_2_fir.join(micro_3_5_fir,"transaction_code")\
                    .join(micro_6_7_fir,"transaction_code")\
                    .join(basket_summ,"transaction_code")


aggregate_all_sec = micro_2_sec.join(micro_3_5_sec,"transaction_code")\
                    .join(micro_6_7_sec,"transaction_code")\
                    .join(basket_summ,"transaction_code")




aggregate_all_2_fir = aggregate_all_fir.withColumn("sm_size",F.when(F.col("cnt_needstate").isin(1,2,3),"VERY_SMALL")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(4,5),"SMALL")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(6,7,8),"MEDIUM")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(9,10,11,12),"LARGE")
                                                  .otherwise(F.when(F.col("cnt_needstate")>12,"VERY_LARGE")
                                                  .otherwise("MISSING"))))))\
                     .withColumn("sm_comp",F.when(F.col("sm_size")=="VERY_SMALL",F.col("dom_micro_2"))
                                .otherwise(F.when(F.col("sm_size")=="SMALL",F.col("dom_micro_3_5"))
                                .otherwise(F.when(F.col("sm_size")=="MEDIUM",F.col("dom_micro_6_7"))
                                .otherwise(""))))\
                     .withColumn("fir_perc",F.when(F.col("sm_size")=="VERY_SMALL",F.col("dom_perc_spend_2"))
                                .otherwise(F.when(F.col("sm_size")=="SMALL",F.col("dom_perc_spend_3_5"))
                                .otherwise(F.when(F.col("sm_size")=="MEDIUM",F.col("dom_perc_spend_6_7"))
                                .otherwise(0))))\
                     .withColumn("sm_concat",F.concat(F.col("sm_size"),F.lit("_"),F.col("sm_comp")))


aggregate_all_2_sec = aggregate_all_sec.withColumn("sm_size",F.when(F.col("cnt_needstate").isin(1,2,3),"VERY_SMALL")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(4,5),"SMALL")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(6,7,8),"MEDIUM")
                                                  .otherwise(F.when(F.col("cnt_needstate").isin(9,10,11,12),"LARGE")
                                                  .otherwise(F.when(F.col("cnt_needstate")>12,"VERY_LARGE")
                                                  .otherwise("MISSING"))))))\
                     .withColumn("sm_comp",F.when(F.col("sm_size")=="VERY_SMALL",F.col("dom_micro_2"))
                                .otherwise(F.when(F.col("sm_size")=="SMALL",F.col("dom_micro_3_5"))
                                .otherwise(F.when(F.col("sm_size")=="MEDIUM",F.col("dom_micro_6_7"))
                                .otherwise(""))))\
                     .withColumn("sec_perc",F.when(F.col("sm_size")=="VERY_SMALL",F.col("dom_perc_spend_2"))
                                .otherwise(F.when(F.col("sm_size")=="SMALL",F.col("dom_perc_spend_3_5"))
                                .otherwise(F.when(F.col("sm_size")=="MEDIUM",F.col("dom_perc_spend_6_7"))
                                .otherwise(0))))\
                     .withColumn("sm_concat",F.concat(F.col("sm_size"),F.lit("_"),F.col("sm_comp")))

aggregate_all_2_sec = aggregate_all_2_sec.withColumnRenamed("sm_comp","sm_comp_sec")

agg_all = aggregate_all_2_fir.select("transaction_code",
                           "sm_size","sm_comp","fir_perc")\
                   .join(aggregate_all_2_sec.select("transaction_code","sec_perc","sm_comp_sec"),
                         "transaction_code","left_outer").na.fill(0)\
                   .withColumn("ratio", F.when(F.col("sec_perc")==0,10)
                               .otherwise((F.col("fir_perc")/F.col("sec_perc"))-1))

agg_all = agg_all.withColumn("fire_rule",F.when(F.col("fir_perc")>=0.4,"yes")\
                             .otherwise(F.when((F.col("fir_perc")>=0.33)&(F.col("ratio")>=0.1),
                                               "yes")
                             .otherwise(F.lit("no"))))

agg_all =agg_all.withColumn("sm_concat",F.when((F.col("fire_rule")=="yes")&(F.col("sm_size").isin(["VERY_SMALL","SMALL","MEDIUM"])),
                                     F.concat(F.col("sm_size"),F.lit("_"),F.col("sm_comp")))
                                .otherwise(F.when((F.col("fire_rule")=="no")&(F.col("sm_size").isin(["VERY_SMALL","SMALL","MEDIUM"])),
                                     F.concat(F.col("sm_size"),F.lit("_"),F.lit("MIXED")))
                                .otherwise(F.when(F.col("sm_size").isin(["LARGE","VERY_LARGE"]),F.col("sm_size"))
                                .otherwise(F.concat(F.col("sm_size"),F.lit("_"),F.lit("MIXED"))))))

agg_all = agg_all.distinct()

trans_to_keep = trans_cat_data.groupBy("transaction_code")\
                                .agg(F.sum("net_spend_amt").cast(IntegerType()).alias("net_spend_amt"))\

agg_all = agg_all.join(trans_to_keep,"transaction_code","inner")

print 'join done' 

#agg_all.write.saveAsTable("agg_all_x5_split_spk",mode='overwrite')

agg_new = agg_all.distinct()

print 'distinct done'

agg_new.repartition(200).write.mode("overwrite").parquet("new_miss_tag_4")


if environment is True:
    sc.stop()