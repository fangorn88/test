
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import cPickle as pickle

from bs4 import BeautifulSoup
from nltk.stem.porter import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances

from gensim.models import Word2Vec


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from nltk.stem.snowball import SnowballStemmer
import zipfile


stemmer = SnowballStemmer('english')





#######################################################################
# 1. read in files
#######################################################################


zf = zipfile.ZipFile('/nfs/science/shared/ipythonNotebooks/satendk/train_homeDepo.csv.zip') # having First.csv zipped file.
train = pd.read_csv(zf.open('train.csv'),encoding="ISO-8859-1")

zf = zipfile.ZipFile('/nfs/science/shared/ipythonNotebooks/satendk/test_homeDepo.csv.zip') # having First.csv zipped file.
test = pd.read_csv(zf.open('test.csv'),encoding="ISO-8859-1")

zf = zipfile.ZipFile('/nfs/science/shared/ipythonNotebooks/satendk/product_descriptions.csv.zip') # having First.csv zipped file.
product_descriptions = pd.read_csv(zf.open('product_descriptions.csv'))

zf = zipfile.ZipFile('/nfs/science/shared/ipythonNotebooks/satendk/attributes.csv.zip') # having First.csv zipped file.
attributes = pd.read_csv(zf.open('attributes.csv'))



####################################################################
# Concat attributes to one attribute per product_uid
####################################################################

attributes.dropna(how="all", inplace=True)

attributes["product_uid"] = attributes["product_uid"].astype(int)

attributes["value"] = attributes["value"].astype(str)


def concate_attrs(attrs):
    """
    attrs is all attributes of the same product_uid
    """
    names = attrs["name"]
    values = attrs["value"]
    pairs  = []
    for n, v in zip(names, values):
        pairs.append(' '.join((n, v)))
    return ' '.join(pairs)

product_attrs = attributes.groupby("product_uid").apply(concate_attrs)


# TIP TIP TIP ::: "RESET_INDEX" to make output og groupby a regular df

product_attrs = product_attrs.reset_index(name="product_attributes")


####################################################################
# Concat + Merge all files
####################################################################

df_all = pd.concat([train, test])

df_all = pd.merge(df_all, product_descriptions, how='left', on='product_uid')

df_all = pd.merge(df_all, product_attrs, how="left", on="product_uid")

df_all['product_attributes'] = df_all['product_attributes'].fillna('')


y = train['relevance']

train_df = df_all



toker = TreebankWordTokenizer()

lemmer = wordnet.WordNetLemmatizer()

def text_preprocessor(x):

    tmp = unicode(x)
    tmp = tmp.lower().replace('blu-ray', 'bluray').replace('wi-fi', 'wifi')
    x_cleaned = tmp.replace('/', ' ').replace('-', ' ').replace('"', '')
    tokens = toker.tokenize(x_cleaned)
    return " ".join([lemmer.lemmatize(z) for z in tokens])





# In[7]:



# lemm description
train_df['desc_stem']  = train_df['product_description'].apply(text_preprocessor)

# lemm title
train_df['title_stem'] = train_df['product_title'].apply(text_preprocessor)

# lemm query
train_df['query_stem'] = train_df['search_term'].apply(text_preprocessor)




# REF ::  http://stackoverflow.com/questions/35281691/scikit-cosine-similarity-vs-pairwise-distances

# from sklearn.feature_extraction.text import TfidfVectorizer

# documents = (
#     "Macbook Pro 15' Silver Gray with Nvidia GPU",
#     "Macbook GPU"    
# )

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0,1])



def calc_cosine_dist(text_a ,text_b, vect):
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]

def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)

tfv_orig = TfidfVectorizer(ngram_range=(1,2), min_df=2)
tfv_stem = TfidfVectorizer(ngram_range=(1,2), min_df=2)
tfv_desc = TfidfVectorizer(ngram_range=(1,2), min_df=2)


# TIP : tfidf vectorizer needs list of texts
# TIP : here text = not onlt searchterm, but also product terms..
#       need to "FIT" / "VOCABULARIZE" them together on common vocab... SO THAT later they are similarly vectorized
#                     adding 2 lists => concatenating their content
#       need to "TRANFORM"/ "VECTORIZE"  them separately on above ... SO THAT later their pariwise sims can be calculated

tfv_orig.fit(
    list(train_df['search_term'].values) + 
    list(train_df['product_title'].values) 
) 
tfv_stem.fit(
    list(train_df['query_stem'].values) + 
    list(train_df['title_stem'].values) 
) 
tfv_desc.fit(
    list(train_df['query_stem'].values) + 
    list(train_df['desc_stem'].values)
) 



cosine_orig = []
cosine_stem = []
cosine_desc = []
set_stem = [] 
for i, row in train_df.iterrows():
    cosine_orig.append(calc_cosine_dist(row['search_term'], row['product_title'], tfv_orig))
    cosine_stem.append(calc_cosine_dist(row['query_stem'], row['title_stem'], tfv_stem))
    cosine_desc.append(calc_cosine_dist(row['query_stem'], row['desc_stem'], tfv_desc))
    set_stem.append(calc_set_intersection(row['query_stem'], row['title_stem']))
train_df['cosine_qt_orig'] = np.round(cosine_orig,3)
train_df['cosine_qt_stem'] = np.round(cosine_stem,3)
train_df['cosine_qd_stem'] = np.round(cosine_desc,3)
train_df['set_qt_stem'] = np.round(set_stem,3)



# In[11]:

fea_tfidfs = train_df[["id","cosine_qt_orig","cosine_qt_stem","cosine_qd_stem","set_qt_stem"]]
                      
fea_tfidfs.to_csv('fea_tfidfs.csv')


# In[ ]:



