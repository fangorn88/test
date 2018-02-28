
# coding: utf-8

# In[46]:

# #Preparartion
import pandas as pd
import time
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import difflib

#----------------------------------------------------------------------------- Parameters
stemmer = SnowballStemmer("english")
distance = 0.2
tesco_range = pd.read_csv("TescoRange.csv", header= 0)
recipe = pd.read_csv("TESCOREALFOODRECIPEONLY.csv", header= 0)
recipe = recipe[pd.isnull(recipe['INGREDIENTS'])== False]
recipe = recipe.reset_index(drop=True)
#Finding the main ingredients
# print test.ix[0,0]
start_time = time.time()
#----------------------------------------------------------------------------- Text cleaning function
# newpoints = np.array([x.split('*') for x in test.ix[0,0]])
# print test.ix[0,0]
# newpoints

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start = start + len(sub)
# star_position= list(find_all(test[0], '*'))
# rowchange_position = list(find_all(test[0], '\n'))   
# print 'Occurence of * in data:', list(find_all(test[0], '*'))
# print 'Occurence of \n in data:', list(find_all(test[0], '\n'))

# print rowchange_position[len(rowchange_position)-1]

op1= []
op2= []
for m in range(len(recipe)):
    star_position= list(find_all(recipe.ix[m,'INGREDIENTS'], '*'))
    rowchange_position = list(find_all(recipe.ix[m, 'INGREDIENTS'], '\n'))
    for t in range(len(star_position)):
        try:
                z= min(x for x in rowchange_position if x > star_position[t] + 1)        
        except ValueError,e:
                z= len(recipe.ix[m,'INGREDIENTS'])
        op1.append(recipe.ix[m,'INGREDIENTS'][star_position[t]: z])
        op2.append(recipe.ix[m,'RECIPENAME'])
    

test=pd.DataFrame(zip(op1, op2))

#----------------------------------------------------------------------------------BasicCleaning functions
    
def text_processor( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    
    # 1. Remove HTML
#     review_text = replacer(raw_review)
    #
    # 2. Remove non-letters        
    review_text = re.sub("[^a-zA-Z0-9]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    review_text1 = review_text.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = stopwords.words("english")    
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in review_text1 if not w in [stops, 'to', 'serve', 'decorate', 'for', 'cut', 'snipped', 'tbsp','extra',
                                                             'virgin', 'everyday', 'value', 'fresh', 'small', 'large', 'medium']] 
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

#-------------------------------------------------------------------------Sorting alphanumeric list
def sort_fn( l ): 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)




x=[]
for i in range(len(test)):
    y=text_processor(str(test.ix[i, 0]))
    x.append(y)
    
cleaned_ingd = pd.DataFrame(zip(op1, op2, x))
cleaned_data = cleaned_ingd.ix[cleaned_ingd.ix[:,2] != cleaned_ingd.ix[0,2], [1,2]]
cleaned_data

#Getting the quantity of ingredients

cleaned_data= cleaned_data.reset_index(drop=True)
for x in range(len(cleaned_data)-1):
    if len(re.findall("[-+]?\d+[\.]?\d*", cleaned_data.ix[x, 2])) >0 :
        cleaned_data.ix[x, 'Weights'] = re.findall("[-+]?\d+[\.]?\d*", cleaned_data.ix[x,2 ])[0] #Extract quantity
        if len(re.findall("[-+]?\d+[\.]?\d*(\w\w)", cleaned_data.ix[x, 2]))>0 :
            cleaned_data.ix[x, 'Unit'] = re.findall("[-+]?\d+[\.]?\d*([\w])","".join(cleaned_data.ix[x,2].split()))[0] #Extract g and m
        else:
            cleaned_data.ix[x, 'Unit'] = 'None'
        
    else:
        cleaned_data.ix[x, 'Weights']=0
        cleaned_data.ix[x, 'Unit'] = 'None'


# g- grams, t- tbps , m- ml , x - to be reviewed manually , None = seperate units(no SI units)

#-------------------------------------------- Cleaning up ingredients further removing weight and stopwords
#---------------------------------------------Probability of main Ingredients by ||recipe name||
#stopwords list
# specific_stopwrords = ['hulled', 'sliced', 'skimmed', 'chopped', 'beaten', 'crushed']
# measures = ['g', 'gms', 'kg', 'tbsp' , 'tsp', 'ml','slices']
for x in range(len(cleaned_data)):
    text= nltk.word_tokenize(cleaned_data.ix[x,2])
    cleaned_data.ix[x,'Cleaned_ingd'] = ' '.join(sort_fn([word for (word, tag) in nltk.pos_tag(text) if tag in ['NN', 'NNS', 'ADJ', 'JJ']]))
    cleaned_data.ix[x, 'Main_ingd_probab'] = difflib.SequenceMatcher(None,cleaned_data.ix[x,'Cleaned_ingd'],cleaned_data.ix[x,1]).ratio()
    cleaned_data.ix[x,'Cleaned_ingd_final'] = ' '.join([stemmer.stem(w) for w in cleaned_data.ix[x,'Cleaned_ingd'].split() if 
                                                        (len(w) > 2 and w not in ['tbsp', 'tsp', 'teaspoon', '100ml',])])

    
cleaned_data.columns = ['RECIPENAME','INGREDIENTS','Weights','Unit','Cleaned_ingd','Main_ingd_probab','Cleaned_ingd_final']

#--------------------------------------------for ingredients in gms
prods_to_match_g = cleaned_data.loc[cleaned_data.ix[:,'Unit']=='g']
#--------------------------------------------for ingredients in ml
prods_to_match_m = cleaned_data.loc[cleaned_data.ix[:,'Unit']=='m']
#--------------------------------------------finding top products
y=pd.DataFrame(prods_to_match_g[['Cleaned_ingd_final','RECIPENAME']].groupby(['Cleaned_ingd_final']).count().sort('RECIPENAME',ascending = False))
y['INGREDIENT'] = y.index.values
y.index = range(len(y.index.values))
#-------------------------------------------------1000 Top ingredients
top_ingredients= y['INGREDIENT'] [0:1000].values
#-------------------------------------------------20 Top Meat Products
top_meat=y[(y['INGREDIENT'].str.contains('meat')) | (y['INGREDIENT'].str.contains('steak'))|(y['INGREDIENT'].str.contains('pork'))|
 (y['INGREDIENT'].str.contains('mutton'))|(y['INGREDIENT'].str.contains('beef'))|(y['INGREDIENT'].str.contains('chicken'))][0:20]


# #--------------------------------------------Matching products to Tescco's current Range
for x in range(len(tesco_range)):
    tesco_range.ix[x,'Clnd_prod_desc']= text_processor(tesco_range.ix[x,'Product'])
    p=len(tesco_range.ix[x,'Clnd_prod_desc'])
    tesco_range.ix[x,'Clnd_prod_quant'] = re.sub("[^0-9]", " ", tesco_range.ix[x,'Clnd_prod_desc'])[p/2 : p].strip()
    tesco_range.ix[x,'Clnd_prod_text'] = re.sub("[^a-zA-Z]", " ", tesco_range.ix[x,'Clnd_prod_desc'])
    text= nltk.word_tokenize(tesco_range.ix[x,'Clnd_prod_text'])
    text_brand = nltk.word_tokenize(tesco_range.ix[x,'Brand'].lower())
    tesco_range.ix[x,'Match String'] = ' '.join(sort_fn([word for word in text if word not in text_brand]))
    tesco_range.ix[x,'Match_String_Final'] = ' '.join([stemmer.stem(w) for w in tesco_range.ix[x,'Match String'].split() if len(w) > 2])
    
#--------------------------------------------for product in gms
tescoprods_to_match_g = tesco_range[(tesco_range['Match String'].str.contains(' g')) & (tesco_range['Brand'] == 'TESCO')]
print("--- RUNTIME : %s seconds ---" % (time.time() - start_time))


#--------------------------------------------Adding the number of people served into cleaned data
for x in range(len(recipe)):    
    recipe.ix[x,'Serving'] = re.sub("[^0-9]", " ", recipe.ix[x,'SERVINGS']).strip()[0:1]
servings_data= pd.merge(cleaned_data, recipe[['RECIPENAME', 'Serving']], on= 'RECIPENAME', how='left')
servings_data.to_csv('servings_data_final.csv')


# In[91]:

#----------------------------------------------------------------Creating series we are matching on:
count_ings = 31
#-----------------------------------------------------------------
# ing = pd.Series(prods_to_match_g['Cleaned_ingd_final'].unique())[30:count_ings]
ing = pd.Series(top_ingredients)[0:50]                                 # Ingredients
prod = pd.Series(tescoprods_to_match_g['Match_String_Final'].unique()) # Tesco products
print "combinations = ", ing.shape[0]*prod.shape[0]
#----------------------------------------------------Merge INGgedients and products, cross join
ings_prods = pd.merge( pd.DataFrame(ing) ,pd.DataFrame(prod) , how='outer')
df_a = pd.DataFrame({"ing":ing})
df_a['x'] = 1
df_b = pd.DataFrame({"prod":prod})
df_b['x'] = 1
ings_prods = pd.merge(df_a, df_b, on='x', how='outer')
# ---------------------------------------------------Handle any missing values
ings_prods['ing'] = ings_prods['ing'].fillna('')
ings_prods['prod'] = ings_prods['prod'].fillna('')

#-------------------------------------------------------------------------Match based on words found
def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b))

start_time = time.time()
set_stem = [] 
for i, row in ings_prods.iterrows():
    set_stem.append(calc_set_intersection(row['ing'], row['prod']))    
ings_prods['set_qt_stem']  = np.round(set_stem,3)
print("--- RUNTIME : %s seconds ---" % (time.time() - start_time))

# 1. Output Top3 productdescs here per Ingredient ::: TO EVALUATE ALGO
# 2. Then merge back to Actual product list to output 2nd & bigger set e.g. multiple packsizes ::: TO DRAW INSIGHTS
# (ings_prods.ix[ings_prods["set_qt_stem"] >0,:]).to_csv('Distance_ing_prod_20_40.csv')
ing_prod_c=ings_prods[ings_prods['set_qt_stem']>0]

#-----------------------------------------------------------------TF-IDF with cosine similarity (From :  Saty)
from sklearn.metrics import pairwise_distances
# REF ::  http://stackoverflow.com/questions/35281691/scikit-cosine-similarity-vs-pairwise-distances

# from sklearn.feature_extraction.text import TfidfVectorizer

# documents = (
#     "Macbook Pro 15' Silver Gray with Nvidia GPU",
#     "Pro Macbook 15 "    
# )

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0,1])



####################################################################################
###### parameter parameter parameter FINE TUNE THIS SETTING 
####################################################################################

tfv_orig = TfidfVectorizer(ngram_range=(1,2))



# TIP : simply concatenate 2 lists or strings by adding "+"
# TIP : create a joint dictionary vector
tfv_orig.fit(
    list(ing_prod_c['ing'].values) + 
    list(ing_prod_c['prod'].values) 
) 
# TIP : to record runtime stats 
start_time = time.time()


def calc_cosine_dist(text_a ,text_b, vect):
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]


cosine_orig = []

for i, row in ing_prod_c.iterrows():
    cosine_orig.append(calc_cosine_dist(row['ing'], row['prod'], tfv_orig))
    
ing_prod_c['cosine_dist'] = np.round(cosine_orig,3)



print("--- RUNTIME : %s seconds ---" % (time.time() - start_time))
print "--- Ingredients * Products ---", ing.shape[0], "*", prod.shape[0], "=", ings_prods.shape[0]

# 1. Output Top3 productdescs here per Ingredient ::: TO EVALUATE ALGO
# 2. Then merge back to Actual product list to output 2nd & bigger set e.g. multiple packsizes ::: TO DRAW INSIGHTS

ing_prod_c.to_csv('Cosine_dis_1000_top_ingredients.csv')



# In[61]:

#--------------------------------------------------------------------------------------------Which product match to what
ing_prod_c.columns
#-----Checking the table (the x number of ingredients that we could capture )
df=ing_prod_c[ing_prod_c['cosine_dist'] < 0.4]   # Distance =0.2 (filtered again later in sas code) 
x= pd.merge(tescoprods_to_match_g, df, left_on = 'Match_String_Final',right_on = 'prod', how='inner')
#-----------------------------Creating product size, product name and ingredients mapped to list
matched_ingredients_final = x[["Product", "Clnd_prod_quant", "ing", "cosine_dist"]]
matched_ingredients_final.to_csv("matched_ingredients.csv")


# In[45]:

# This code requires output from SAS. Used for creating lists within dataframe
#------This CSV contains the matches where distance < 0.2 and sales of tesco product >10000
matched_ingredients_final= pd.read_csv("valid_mtchd_ingd.csv", header= 0)
ldf=matched_ingredients_final
ldf= ldf.sort(['ing'], ascending= True)
ldf= ldf.reset_index()
op2=[]
for x in range(len(matched_ingredients_final)-1):
    
    if ldf.ix[x+1, "ing"] == ldf.ix[x, "ing"]:
        op2.append(ldf.ix[x,'clnd_prod_quant'])
    else:
        op2.append(ldf.ix[x,'clnd_prod_quant'])
        ldf.ix[x,"clnd_prod_quant" ] = str(op2)
        ldf.ix[x, "index"] = "xxx"
        op2= []
ldf

ingd_tesco_pck_size = ldf[ldf["index"]== 'xxx'].ix[:, ["clnd_prod_quant","ing"]]
ingd_tesco_pck_size.to_csv("matchd_pck_sizes.csv") # Read this file in SAS to get the table "ingd_innov_oppor"


# In[14]:

#---------------------------------------------------------------Sequence Matcher || ----NOT BEING USED NOW----
match_rate= 0.8
#---------------------------------------------------------------
start_time = time.time()
print tescoprods_to_match_g.columns
df_a = pd.DataFrame(top_meat['INGREDIENT'])
df_a['x'] = 1
df_b = pd.DataFrame(tescoprods_to_match_g.ix[:,'Match_String_Final'].unique())
df_b['x'] = 1
df = pd.merge(df_a, df_b, on='x', how='outer')
for x in range(len(df)):
    df.ix[x,'Match Rate'] = difflib.SequenceMatcher(None,df.ix[x,'INGREDIENT'],df.ix[x,2]).ratio()
match_result = df[df['Match Rate']> match_rate]
match_result.columns= ['Ingredient', 'x', 'Prod', 'Match_Rate']
print("--- RUNTIME : %s seconds ---" % (time.time() - start_time))


# In[359]:

df_a = cleaned_data
df_a['x'] = 1
df_b = recipe
df_b['x'] = 1
df = pd.merge(df_a, df_b, on='x', how='outer')
df


# In[433]:

x= cleaned_data[cleaned_data['Cleaned_ingd_final'].isin(ingredients.tolist())]
x= x[x['Unit'] == 'g']
x


# In[19]:

ldf.ix[1,"Clnd_prod_quant"] = str([100, 200, 300])
ldf


# In[97]:

servings_data

