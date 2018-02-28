"""
    Need States
"""
import logging
import re  # Regular expression operations
import ast # Abstract Syntax Trees
import json
import sys
import warnings
import importlib
import getpass
from time import time
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark import SparkContext
from pyspark.sql import HiveContext
import configparser
from scipy.cluster.hierarchy import dendrogram, linkage


sc = SparkContext()
sqlContext = HiveContext(sc)

# some setting for this notebook to actually show the graphs inline, you probably won't need this
'''matplotlib.use('PDF')
pd.options.display.float_format = '{:,.2f}'.format
try:
    %matplotlib inline
except:
    pass
'''

#######################################################
# confireader
#######################################################
def parse_config(old_config):
    """
    Parse the config file to parse config into json
    """
    new_config = {}
    for sec in old_config.sections():
        new_config[sec] = {}
        for key in old_config[sec].keys():
            key_val = old_config[sec][key]
            try:
                new_config[sec][key] = ast.literal_eval(key_val)
            except StandardError:
                if key_val == "":
                    key_val = None
                elif key_val[0] in ["[", "{", "("] or key_val[-1] in ["]", "}", ")"]:
                    err_message = "Value cannot be parsed for variable: \
{0} of the Section: {1}.\nSyntax Error in value:{2}".format(key, sec, key_val)
                    raise Exception(err_message)
                new_config[sec][key] = key_val
    return new_config


#######################################################
# Funtion to set spark context configuration
#######################################################
def check_env():
    """
    Check for spark environment 
    """
    job_file = [
        x for x in sys.argv if x.endswith('py') and not x.endswith('__main__.py')]
    spark_submit = True if len(job_file) > 0 else False
    return spark_submit

def spark_config_set(is_spark_submit):
    """
    Set the spark submit global variables
    """
    if is_spark_submit:
        global sc, sqlContext
        sc = SparkContext()
        sqlContext = HiveContext(sc)
        
def spark_config_reset(is_spark_submit):
    """
    Reset the spark submit global variables
    """
    if is_spark_submit:
        sc.stop()
    print "End of the job!"
    
def spark_config():
    """
    Set hive context configurations
    """
    sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
    sqlContext.sql("SET spark.sql.parquet.compression.codec=snappy")
    sqlContext.setConf('spark.sql.hive.convertMetastoreParquet', 'False')
    sqlContext.clearCache()
    # print(sc._conf.get("spark.yarn.queue"))
    
##########################################################
# Transform result into permanent table or csv file
##########################################################
def from_df_to_hdfs(f_output_table,
                    f_out_file,
                    f_path=None):
    """
    Save dataframe to csv file or database table
    """
    if f_path is None:
        f_path = "/user/{0}/".format(get_user())
    if f_out_file.split(".")[-1] == "csv":
        f_output_table.write.save(
            path= f_path + f_out_file,
            mode="overwrite",
            format="com.databricks.spark.csv",
            header="true")
    else:
        f_output_table.write.saveAsTable(f_out_file, mode="overwrite")

########################################################
# Identification of required list is empty or not
########################################################
def is_empty(sample_list=None):
    """
    If list is none or empty, need to be reset to empty list

    Usage:
    >>> is_empty(None)
    []
    >>> is_empty(["abc"])
    ['abc']
    >>> is_empty([])
    []
    """
    if sample_list is None or len(sample_list) == 0:
        sample_list = []
    return sample_list

        
#############################################################
# Converting list of strings to lowercase list of strings        
#############################################################
def to_lower(column_list):
    """
    Convert list of strings to list of lowercase string

    Usage:
    >>> to_lower(["hello", "Hello", "HELLO", "hElLo"])
    ['hello', 'hello', 'hello', 'hello']
    """
    return [col.lower() for col in column_list]


##############################################################
# Converting string or floating datatype to integers
##############################################################
def to_int(num):
    """
    Evaluating number is encoded within specified format and typecast to int
        int, float, long or number string

    Usage:
    >>> to_int("100")
    100
    >>> to_int("70.2")
    70
    >>> to_int(35.2)
    35
    """
#     ast.literal_eval raises an exception if the input isn't
#     a valid Python datatype, so the code won't be executed if it's not.
    if isinstance(num, basestring):
        try:
            num = ast.literal_eval(num)
        except Exception:
            err_message = "ValueError: Invalid value, must be integer or float"
            raise Exception(err_message)
    if not isinstance(num, int):
        try:
            num = int(num)
        except Exception:
            err_message = "ValueError: Invalid datatype, must be integer or float"
            raise Exception(err_message)
    if num < 1:
        err_message = "ValueError: Invalid value, must be positive integer"
        raise Exception(err_message)
    return num

##########################################################
# Calcutate median based on tiles results
##########################################################
def median(tile_1,
           tile_2,
           count):
    """
    Calculate median based on number of records:
        if even number of records then average else tile 1 value
    """
    if count % 2 != 0:
        return float(tile_1)
    else:
        return (float(tile_1) + float(tile_2)) / 2.0
    
##########################################################  
# Get the user information
##########################################################
def get_user():
    """
    Return name of current user
    """
    return getpass.getuser()


################################################################ 
# Convert product category level to equivalent product columns
################################################################ 
def get_prod_level(prod_cat_level):
    """
    Covert product category level to equivalent product columns
    """
    if prod_cat_level > 0:
        prod_l_code = "prod_hier_l{0}_code".format(prod_cat_level)
        prod_l_desc = "prod_hier_l{0}_desc".format(prod_cat_level)
    else:
        prod_l_code = "generaltype_id"
        prod_l_desc = "category_name"
    prod_l_cat = "romir_category"
    return prod_l_code, prod_l_desc, prod_l_cat


###############################################################
# Replace set of characters with the required character
###############################################################
def replace_chars(field, esc_chars, rep_ch):
    """
    Replace set of characters with the required character in dataframe
    """
    res_field = "P"
    if field is not None:
        res_field = re.sub(esc_chars, rep_ch, field).upper()
        # res_field = "".join([rep_ch if ch in esc_chars else ch for ch in field.strip()])
    return res_field

###############################################################
# Convert dataframe or csv to pandas dataframe
###############################################################
def df_to_pandas(df_name):
    """
    Convert dataframe or csv to pandas dataframe
    """
    if isinstance(df_name, basestring):
        p_df = pd.read_csv(df_name)
        print "Creating dataframe from file"
    else:
        p_df = df_name.toPandas()
        print "Creating dataframe from df_name"

    # ******************** MUKUL ********************
    # print "Column names are: ",p_df.columns.values
    # p_df = p_df.reset_index()
    # print "Column names after reset index are: ",p_df.columns.values
    # ***********************************************

    p_df = p_df.set_index('category')
    return p_df


###########################################################
# Convert distance matrix to squareform matrix
###########################################################
def matrix_to_squareform(distance_matrix):
    """
    Convert distance matrix to squareform matrix
    """
    final_matrix = []
    for index, rows in enumerate(distance_matrix):
        try:
            final_matrix.extend(rows[index+1:])
        except Exception as e:
            # print "Failed at index:",index,e
            return np.array(final_matrix)
    return np.array(final_matrix)

######################################################################################################################
# Convert distance matrix to Algomerative hierarical clustering using scipy linkage with specified method
######################################################################################################################
def hierarical_clustering(p_df, method="average"):
    """
    Convert distance matrix to Algomerative hierarical clustering
    using scipy linkage with specified method
    """
    pdf_values = p_df.values
    np.fill_diagonal(pdf_values, 0)
    pdf_values_1_d = matrix_to_squareform(pdf_values)
    cluster_matrix = linkage(pdf_values_1_d, method)
    return cluster_matrix


###########################################################################
# Convert dataframe to cluster matrix and save result in csv
###########################################################################
def table_to_csv(output_table, cat_column, method, out_csv_names, debug):
    """
    Convert dataframe to cluster matrix and save result in csv based on debug mode
    convert cluster matrix to excel macro compatible csv format
    """
    p_df = df_to_pandas(output_table)
    no_of_prod = len(p_df)
    head_df = pd.DataFrame()
    head_df["Cluster Name"] = p_df.reset_index()[cat_column]
    head_df_list = head_df["Cluster Name"].tolist()
    try:
        cluster_matrix = hierarical_clustering(p_df, method)
    except Exception as e:
        raise Exception("Distance matrix has some issue:"+str(e))
    # head_df.sort("Cluster Name", inplace=True)    # original
    head_df = head_df.sort_values(["Cluster Name"]) # changed by mukul
    head_df["Cluster Number"] = range(1, no_of_prod + 1)
    head_df = change_column_order(head_df, "Cluster Number", 0)
    p_df = pd.DataFrame(cluster_matrix, columns=["Idj1", "Idj2", "SemipartialRSq", "priority"])
    p_df["NumberOfClusters"] = range(len(p_df),0,-1)
    p_df = format_column(p_df, "Idj1", no_of_prod, "NumberOfClusters")
    p_df = format_column(p_df, "Idj2", no_of_prod, "NumberOfClusters") 
    p_df.drop("priority", axis=1, inplace=True)
    p_df = change_column_order(p_df, "NumberOfClusters", 0)
    if not debug:
        p_df.to_excel(out_csv_names[0], index=False)
        head_df.to_excel(out_csv_names[1], index=False)
    return head_df, p_df, head_df_list, cluster_matrix


##########################################################################
# Display dendogram from the cluster matrix
##########################################################################
'''
def cluster_to_dendogram(p_df_list, cluster_matrix):
    """
    Display dendogram from the cluster matrix
    """
    plt.figure(figsize=(6, 10))
    plt.title('Dendrogram')
    plt.xlabel('distance')
    dendrogram_ret_args = dendrogram(
        cluster_matrix,
        # leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        labels = p_df_list,
        orientation='right'
    )
    plt.show()
    return dendrogram_ret_args['ivl'][::-1]
'''
##########################################################################
# Format Column based on Cluster index in cluster matrix
##########################################################################

def format_column(p_df, column, no_of_prod, idx_col):
    """
    Format Column based on Cluster index in cluster matrix
    """
    p_df[column] = p_df[column].astype(int) + 1
    p_df[column] = np.where(p_df[column] <= no_of_prod,
                            p_df[column].astype(str),
                            "CL" + p_df[idx_col][p_df[column]-no_of_prod-1].astype(str)
                                )
    p_df[column] = p_df[column].map(lambda x: x.rsplit('.', 1)[0])
    return p_df


##########################################################################
# Change the position of a column in pandas dataframe
##########################################################################
def change_column_order(p_df, col_name, index):
    """
    Change the position of a column in pandas dataframe
    """
    cols = p_df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return p_df[cols]

##########################################################################
# Create dataframe from hive table
##########################################################################
def table_to_df(db_name, table_name):
    """
    Create dataframe from hive table
    """
    return sqlContext.table("{0}.{1}".format(db_name, table_name))

##########################################################################
# Fetch product table with the required category level data with the index
##########################################################################
def get_prod_table_indexed(prod_table,
                           prod_l_code,
                           prod_l_desc,
                           prod_l_plus_desc,
                           prod_l_cat):
    """
    Fetch product table with the required category level data with the index , prod_l_cat is the new column that is created (passed as a parameter)
    """
    
    char_replace = F.udf(replace_chars, F.StringType()) #udf will run row by row on partition dataframe, but normal full will run on dataframe  # * will serialise the list , ** will serialise the dictionary, krags,kwargs
    if prod_l_plus_desc in trans.columns:
        prod_columns = [prod_l_code, prod_l_desc, prod_l_plus_desc]
        return prod_table.select(prod_columns).withColumn(
            prod_l_cat,
            F.concat(F.lit("_"),
                     char_replace(prod_l_code, F.lit(r"[^A-Za-z0-9]+"), F.lit(r"")),
                     F.lit("_"),
                     char_replace(prod_l_desc, F.lit(r"[^A-Za-z]+"), F.lit(r"")),
                     F.lit("_"),
                     char_replace(prod_l_plus_desc, F.lit(r"[^A-Za-z]+"), F.lit(r""))
                     )
        ).withColumn(
            "prod_index",
            F.dense_rank().over(Window.orderBy(prod_l_cat)))
    
    else:
        prod_columns = [prod_l_code, prod_l_desc]
        return prod_table.select(prod_columns).withColumn(
            prod_l_cat,
            F.concat(F.lit("_"),
                     char_replace(prod_l_code, F.lit(r"[^A-Za-z0-9]+"), F.lit(r"")),
                     F.lit("_"),
                     char_replace(prod_l_desc, F.lit(r"[^A-Za-z]+"), F.lit(r""))
                    )
        ).withColumn(
            "prod_index",
            F.dense_rank().over(Window.orderBy(prod_l_cat)))

##########################################################################
# Calculate total number of unique transactions in transaction table
##########################################################################
def get_total_trans(all_customers_data, trans_column):
    """
    Calculate total number of unique transactions in transaction table
    """
    return all_customers_data.select(trans_column).distinct().count()

##########################################################################
# Calculate total number of unique transactions in each category
##########################################################################
def get_grouped_prod(all_customers_data, trans_column, prod_l_cat):
    """
    Calculate total number of unique transactions in each category
    """
    return all_customers_data.select(trans_column, prod_l_cat)\
.groupBy(prod_l_cat).agg(F.countDistinct(trans_column).alias('hhds'))

##########################################################################
##########################################################################
# Calculate total number of unique transactions in combination of categories(important)
##########################################################################
##########################################################################
def get_grouped_both(all_customers_data, trans_column, prod_l_cat):
    """
    Calculate total number of unique transactions in combination of categories
    """
    return all_customers_data.alias("p").join(
            all_customers_data.alias("q"),
            trans_column
        ).select(
            F.col("p.{0}".format(prod_l_cat)).alias("category_a"),
            F.col("q.{0}".format(prod_l_cat)).alias("category_b"),
            trans_column
        ).filter(
            "category_a != category_b"
        ).groupBy(
            "category_a", "category_b"
        ).agg(
            F.countDistinct(trans_column).alias('hhds_both')
        )

#############################################################################
# Calculate all the measures required to calculate chi square value
#############################################################################
def get_grouped_meas(grouped_both, grouped_prod, total_trans, prod_l_cat):
    """
    Calculate all the measures required to calculate chi square value
        Join total uique transactions table
        Join total uique transactions per category table
        Join total uique transactions per group of category table
    """
    return grouped_both.alias("ab").join(
            grouped_prod.select(
                F.col(prod_l_cat).alias("category_a"),
                F.col("hhds").alias("hhds_a")).alias("ga"),
            "category_a"
            ).join(
            grouped_prod.select(
                F.col(prod_l_cat).alias("category_b"),
                F.col("hhds").alias("hhds_b")).alias("gb"), 
            "category_b"
            ).withColumn(
                "hhds_all",
                F.lit(total_trans)
            )

#############################################################################
# Calculation of expected, chi square and partial index from measures table
#############################################################################
def get_grouped_chi(grouped_meas):
    """
    Calculation of expected, chi square and partial index from measures table
    """
    return grouped_meas.withColumn(
                "exp_both",
                F.col("hhds_a") * F.col("hhds_b") / F.col("hhds_all")
            ).withColumn(
                "chiab",
                F.when(
                    F.col("exp_both") != 0,
                    F.pow(F.col("hhds_both") - F.col("exp_both"), 2) / F.col("exp_both")
                ).otherwise(0)
            ).withColumn(
                "partab",F.when(
                    F.col("exp_both") != 0,
                    F.col("hhds_both") / F.col("exp_both")
                ).otherwise(0)
            )

#############################################################################
# Calculation of median index per category from partial index
#############################################################################
def get_median_index(grouped_chi,
                     median_col,
                     partition_col):
    """
    Calculation of median index per category from partial index
    """
    median_udf = F.udf(
        median,
        T.FloatType()
    )
    
    return grouped_chi.select(
        partition_col,
        median_col
    ).withColumn(
        'ntile',
        F.ntile(2).over(
            Window().partitionBy(
                partition_col
            ).orderBy(
                F.col(
                    median_col
                ).desc()
            )
        )
    ).groupBy(
        partition_col,
        'ntile'
    ).agg(
        F.count(F.col(median_col)).alias('ntile_count'),
        F.max(F.col(median_col)).alias('ntile_max'),
        F.min(F.col(median_col)).alias('ntile_min')
    ).groupBy(
        partition_col
    ).agg(
        F.min(
            'ntile_max'
        ).alias(
            '1st_tile'
        ),
        F.max(
            'ntile_min'
        ).alias(
            '2nd_tile'
        ),
        F.sum(
            'ntile_count'
        ).alias(
            'count'
        )
    ).select(
        partition_col,
        median_udf(
            F.col('1st_tile'),
            F.col('2nd_tile'),
            F.col('count')
        ).alias(
            'median_index'
        )
    )


################################################################################
# Calculation of sub index per category from partial and median index
################################################################################

def get_grouped_sub_chi(grouped_chi,
                        median_index,
                        median_col,
                        join_col):
    """
    Calculation of sub index per category from partial and median index
    """
    return grouped_chi.join(
        median_index,
        join_col
    ).withColumn(
        "subab",
        F.col(median_col) / F.col("median_index")
    ).filter(
        F.col("subab")>1
    ).withColumn(
        "chiperc",
        F.col("chiab") * 100 / F.sum(
            "chiab"
        ).over(Window.partitionBy("category_a"))
    )


###################################################################################
# Calculation of chi square percentage and distance from chi measures table
###################################################################################
def get_grouped_distance(grouped_sub_chi,
                         partition_col,
                         threshold):
    """
    Calculation of chi square percentage and distance from chi measures table
    """
    grouped_sub_chi_t1 = grouped_sub_chi.withColumnRenamed(
        "chiperc",
        "t1_chiperc"
    )
    grouped_sub_chi_t2 = grouped_sub_chi.select(
        F.col("category_a").alias("category_b"),
        F.col("category_b").alias("category_a"),
        F.col("chiperc").alias("t2_chiperc")
    )
    return grouped_sub_chi_t1.join(
        grouped_sub_chi_t2,
        ["category_a", "category_b"],
        "outer"
    ).withColumn(
        "distance",
        F.when(
            (F.col("t1_chiperc") > 0) & (F.col("t2_chiperc") > 0),
            1 / ((F.col("t1_chiperc") + F.col("t2_chiperc")) / 2)
        ).otherwise(
            F.when(F.col("t1_chiperc") > 0,
                   1 / F.col("t1_chiperc")
                  ).otherwise(
                F.when(F.col("t2_chiperc") > 0,
                       1 / F.col("t2_chiperc")
                      ).otherwise(0)
            )
        )
    ).withColumn(
        "rank",
        F.dense_rank().over(
            Window.partitionBy(
                partition_col
            ).orderBy(
                "distance"
            )
        )
    ).withColumn(
        "n_tile",
        F.ntile(100).over(
            Window.partitionBy(
                partition_col
            ).orderBy(
                "distance"
            )
        )
    ).filter(
        (F.col("n_tile") <= threshold) & (F.col("distance") <= 1000)
    ).orderBy(
        "category_a",
        "distance"
    )


##############################################################
# Coverting dataframe to distance matrix using pivot
##############################################################
def get_distance_matrix(grouped_distance):
    """
    Coverting dataframe to distance matrix using pivot
    """
    return grouped_distance.groupby(
        F.col(
            "category_a"
        ).alias(
            "category"
        )
    ).pivot(
        "category_b"
    ).agg(
        F.expr(
            "coalesce(min(distance), 10000.00)"
        )
    ).orderBy(
        "category"
    )


start_time = time()
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()
debug             = False
trans_column      = "basket_id"
prod_cat_level    = None
prod_l_plus_desc  = None
out_file          = ["distance_average_nsovr1.xlsx", "cluster_average_nsovr1.xlsx", "dendogram_index_list_nsovr1.xlsx"]
temp_time = time()
# prod_l_code, prod_l_desc, prod_l_cat = get_prod_level(prod_cat_level)
prod_l_code = "product_code"
prod_l_desc = "ns_harm"
prod_l_cat  = "ns_harm" #variable that will have values concat of code and desc

print prod_l_code
print prod_l_desc
print prod_l_cat

user = get_user()
print "user: ",user


char_replace   = F.udf(replace_chars, F.StringType())

# trans          = sqlContext.sql("""select * from x5_ru_analysis.kg_SM_karusel_100per2_52weeks
#                                     where item_spend>0 and item_qty >0""")
#trans          = trans.withColumnRenamed("category_id","generaltype_id")
# trans          = trans.where(F.col("store_format")=="TRADITIONAL_TRADE")

# Select distinct product_code, dunn_cat_english_1, group_code
#                                   from x5_ru_analysis.ak_prod_karusel_2
# prod_table     = sqlContext.sql("""
# SELECT DISTINCT PRODUCT_CODE,NS_HARM,NS_HARM AS GROUP_CODE FROM (SELECT A.*,CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH) AS NEW_CAT 
# FROM x5_ru_analysis.ak_prod_karusel_2 A) A INNER JOIN 
# X5_RU_ANALYSIS.AK_NS_MAP_KARUSEL1 B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL
#                                   """)

# prod_table  = prod_table.select([prod_l_code,prod_l_desc,"group_code"]).withColumn(
#                 prod_l_cat,
#                 F.concat(F.lit("_"),
#                      "group_code",
#                      F.lit("_"),
#                      char_replace(prod_l_desc, F.lit(r"[^A-Za-z]+"), F.lit(r""))
#                     )
#                     )

bask1 = sqlContext.sql(""" 
SELECT DISTINCT BASKET_ID,
       NS_HARM 
FROM x5_ru_analysis.kg_sm_pyt_52weeks A
INNER JOIN
  (SELECT A.*,
          B.NS_HARM
   FROM
     (SELECT A.*,
             CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH_1) AS NEW_CAT
      FROM x5_ru_analysis.AK_PROD_OVR_1 A) A
   INNER JOIN X5_RU_ANALYSIS.AK_NS_MAP_OVR B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL) B ON A.PRODUCT_CODE = B.PRODUCT_CODE
WHERE item_spend > 0
  AND item_qty > 0
""")

bask2 = sqlContext.sql(""" 
SELECT DISTINCT BASKET_ID,
       NS_HARM 
FROM x5_ru_analysis.kg_sm_karusel_100per2_52weeks A
INNER JOIN (SELECT A.*,B.NS_HARM FROM (SELECT A.*,CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH_1) AS NEW_CAT 
FROM x5_ru_analysis.AK_PROD_OVR_1 A) A INNER JOIN 
X5_RU_ANALYSIS.AK_NS_MAP_OVR B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL ) B 
ON A.PRODUCT_CODE = B.PRODUCT_CODE
WHERE item_spend > 0
  AND item_qty > 0

""")

bask3 = sqlContext.sql(""" 
SELECT DISTINCT BASKET_ID,
       NS_HARM 
FROM x5_ru_analysis.kg_sm_perekrestok_20per_52weeks A
INNER JOIN (SELECT A.*,B.NS_HARM FROM (SELECT A.*,CONCAT('_',GROUP_CODE,'_',DUNN_CAT_ENGLISH_1) AS NEW_CAT 
FROM x5_ru_analysis.AK_PROD_OVR_1 A) A INNER JOIN 
X5_RU_ANALYSIS.AK_NS_MAP_OVR B ON A.NEW_CAT = B.KARU_CATEGORY_FINAL ) B 
ON A.PRODUCT_CODE = B.PRODUCT_CODE
WHERE item_spend > 0
  AND item_qty > 0

""")

df_concat1 = bask1.unionAll(bask2)
trans1 = df_concat1.unionAll(bask3)

# prod_table  = prod_table.select([prod_l_code, prod_l_desc]).withColumn(
#                 prod_l_cat,char_replace(prod_l_desc, F.lit(r"[^A-Za-z]+"), F.lit(r"")))

# all_customers_data = trans.select(trans_column,prod_l_code,prod_l_desc).distinct().withColumn(
#                         prod_l_cat,char_replace(prod_l_desc, F.lit(r"[^A-Za-z]+"), F.lit(r"")))
#                             .join(prod_table,["cat_id","cat_desc"],"left")\
#                             .select(trans_column,prod_l_cat)  #5515953

# all_customers_data = trans.select(trans_column,prod_l_code).distinct().join(prod_table,[prod_l_code],"left")\
#                         .select(trans_column,prod_l_cat).distinct()  #5515953

all_customers_data = trans1.distinct()


# Computation of total number of unique baskets        
total_trans = get_total_trans(all_customers_data, trans_column)
logger.info("total_trans is computed, Value: %s", total_trans)

# Computation of total number of unique baskets for each category
grouped_prod = get_grouped_prod(all_customers_data, trans_column, prod_l_cat)
logger.info("grouped_prod is validated, Value: %s", grouped_prod)

# Computation of total number of unique baskets for each combination of category
# A and B with same basket products
grouped_both = get_grouped_both(all_customers_data, trans_column, prod_l_cat)
logger.info("grouped_both is validated, Value: %s", grouped_both)


# Computation of total number of unique baskets for each combination of category
# A and B without same basket products
grouped_meas = get_grouped_meas(grouped_both, grouped_prod, total_trans, prod_l_cat)
logger.info("grouped_meas is validated, Value: %s", grouped_meas)

# nss_all.write.mode("overwrite").parquet("nss_all.parquet")
# nss_all = sqlContext.read.parquet("nss_all.parquet")

if debug:
        try:
            grouped_meas = sqlContext.read.parquet("/user/{0}/intermediate_grouped_meas".format(user))
        except:
            pass
else:
    grouped_meas.write.parquet("/user/{0}/intermediate_grouped_meas".format(user),
                               mode="overwrite")
    grouped_meas = sqlContext.read.parquet("/user/{0}/intermediate_grouped_meas".format(user))
logger.info("No. of rows in grouped_meas is : %s", grouped_meas.count())


grouped_chi = get_grouped_chi(grouped_meas)
logger.info("grouped_chi is validated, Value: %s", grouped_chi)
    


median_index = get_median_index(grouped_chi, "partab", "category_a")
logger.info("median_index is validated, Value: %s", median_index)



grouped_sub_chi = get_grouped_sub_chi(grouped_chi, median_index, "partab", "category_a")
logger.info("grouped_sub_chi is validated, Value: %s", grouped_sub_chi)

        
grouped_sub_chi.write.saveAsTable("x5_ru_segmentations.chi_sq_score_f",
                                   mode='overwrite')


grouped_distance = get_grouped_distance(grouped_sub_chi, "category_a", 95)
logger.info("grouped_distnace is validated, Value: %s", grouped_distance)


distance_matrix = get_distance_matrix(grouped_distance)
logger.info("distance_matrix is validated, Value: %s", distance_matrix)


# distance_matrix.show(,truncate=False)    


output_table    = distance_matrix

if debug:
    #     doctest.testmod()
        print output_table
        head_df, p_df, head_df_list, cluster_matrix = table_to_csv(
                output_table, "category", "average", out_file, debug)
        print head_df
        print p_df
else:
    
    head_df, p_df, head_df_list, cluster_matrix = table_to_csv(
            output_table, "category", "average", out_file, debug)
'''    dendogram_index_list = cluster_to_dendogram(head_df_list, cluster_matrix)
    pd.DataFrame(dendogram_index_list,
                 columns=["Cluster Name"]).to_csv(out_file[2], index=False) '''
end_time = round((time() - start_time) / 60, 2)


# trans.select(trans_column,prod_l_code).distinct().count()  #5515953

# prod_table_indexed = get_prod_table_indexed(prod_table, prod_l_code,prod_l_desc, prod_l_plus_desc, prod_l_cat)
# prod_table_indexed.show()
# total_trans = get_total_trans(all_customers_data, trans_column)

end_time = round((time() - start_time) / 60, 2)
logger.info("Code is completed successfully in: %s minutes", end_time)