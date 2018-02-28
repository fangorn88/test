
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

from math import sin, cos, sqrt, atan2, radians, degrees, fabs
import string as str
import re
from random import randint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection 
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture


import xgboost as xgb
from nltk.stem import PorterStemmer


# In[2]:

#reading data
train_df = pd.read_json("train.json")
test_df = pd.read_json("test.json")

print 'Files imported'

#creating test_train and y_train
y_train = train_df.interest_level
test_df.interest_level = -1
train_test = pd.concat([train_df,test_df])

y_train_dummies = pd.get_dummies(y_train, prefix = 'interest')
emp_bayes = pd.read_csv('emp_bayes.csv').set_index(train_test.index)


# In[3]:

print 'Creating features' 

features_to_use= []

#clipping outliers
train_test['bathrooms'] = train_test.bathrooms.clip_upper(5)
train_test['bedrooms'] = train_test.bedrooms.clip_upper(5)
train_test['price'] = train_test.price.clip_upper(20000)

features_to_use.extend(['bathrooms','bedrooms','price'])

# Composite features based on: 
# https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths

train_test['num_priceXroom'] = (train_test.price / (1 + train_test.bedrooms.clip(1, 4) + 0.5*train_test.bathrooms.clip(0, 2))).values
features_to_use.append('num_priceXroom')


# In[4]:

#empirical bayes
train_test = pd.merge(train_test,emp_bayes, left_index=True, right_index=True)
features_to_use.extend(emp_bayes.columns)


# In[5]:

# count of photos 
train_test["num_photos"] = train_test["photos"].apply(len)
features_to_use.append('num_photos')

# is bulding id present or not
train_test['building_id_present'] = train_test['building_id'].apply(lambda x: 0 if x == '0' else 1)
features_to_use.append('building_id_present')

# #stores which have listing higher than 7235000 always have low interest
train_test['bad_listing'] = train_test['listing_id'].apply(lambda x: 1 if x>7235000 else 0)
features_to_use.append('bad_listing')


# In[6]:

#importing clean features
features_dup = pd.read_csv("feature_duplicate.csv")

#creating a dictinary of deduplicated features
features = train_test[["features"]].apply(lambda _: [list(map(str.strip, map(str.lower, x))) for x in _])
features_dict = features_dup.set_index('original_feature')['unique_feature'].to_dict()

def features_map_func(x):
    temp_list = []
    for i in x:
        if i in features_dict.keys():
            temp_list.append(features_dict[i])
    return temp_list

#cleaning up features
train_test['new_features'] = features.features.apply(lambda x : features_map_func(x))
train_test['new_features'] = train_test["new_features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

#count vectorizing features
count_vec = CountVectorizer(stop_words='english', max_features=100)

train_test_count_vec_feat = pd.DataFrame(count_vec.fit_transform(train_test["new_features"]).todense(), columns = [s + '_feat' for s in count_vec.vocabulary_.keys()] ).set_index(train_test.index)
train_test = train_test.merge(train_test_count_vec_feat,how='left', left_index = True, right_index = True)

features_to_use.extend([s + '_feat' for s in count_vec.vocabulary_.keys()] )


# In[7]:

#clustering neighbourhood

lat_long = train_test[train_test.longitude> -74.05][train_test.longitude< -73.875][train_test.latitude> 40.63][train_test.latitude< 40.87]
cluster = lat_long[['latitude','longitude']]

model_gm = GaussianMixture(n_components=40, covariance_type='full',tol = 0.01, max_iter=5000, random_state=7, verbose=0)
pred_gm = pd.DataFrame(model_gm.fit(cluster).predict(cluster)).set_index(cluster.index)
pred_gm.columns = ['pred_gm']


train_test = pd.merge(train_test, pred_gm, how = 'left', left_index=True, right_index=True)
train_test.pred_gm[train_test.pred_gm.isnull()] = -1

dummy_neighbourhood = pd.get_dummies(train_test.pred_gm, prefix = 'dummy_nb_')

train_test = train_test.merge(dummy_neighbourhood, how='left', left_index = True, right_index = True)

features_to_use.extend(dummy_neighbourhood.columns)


# In[8]:

### Calculate the distance of all latitude & longitude from the city center 

# approximate radius of earth in km
R = 6373.0

location_dict = {
'manhatten_loc' : [40.7527, -73.9943],
'brooklyn_loc' : [45.0761,-73.9442],
'bronx_loc' : [40.8448,-73.8648],
'queens_loc' : [40.7282,-73.7949],
'staten_loc' : [40.5795,-74.1502]}

for location in location_dict.keys():

    lat1 = train_test['latitude'].apply(radians)
    lon1 = train_test['longitude'].apply(radians)
    lat2 = radians(location_dict[location][0])
    lon2 = radians(location_dict[location][1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    def power(x):
        return x**2

    a = (dlat/2).apply(sin).apply(power) + lat1.apply(cos) * cos(lat2) * (dlon/2).apply(sin).apply(power)
    c = 2 * a.apply(sqrt).apply(sin)

    ### Add a new column called distance
    train_test['distance_' + location] = R * c
    features_to_use.append('distance_' + location)
    
    x = lat1.apply(cos)*sin(lat2) - lat1.apply(sin) * cos(lat2) * (lon2 - lon1).apply(cos)
    y = (lon2 - lon1).apply(sin)* cos(lat2)

    ### Create a new colum as degrees
    train_test['degrees_' + location] = (np.arctan2(x,y)).apply(degrees).apply(fabs)


# In[9]:

#unique buildings managers and addresses

def find_objects_with_only_one_record(feature_name):
    temp = train_test[feature_name].reset_index()
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]

managers_with_one_lot = find_objects_with_only_one_record('manager_id')
buildings_with_one_lot = find_objects_with_only_one_record('building_id')
addresses_with_one_lot = find_objects_with_only_one_record('display_address')

train_test['manager_id_unique'] = 0
train_test['building_id_unique'] = 0
train_test['display_address_unique'] = 0

train_test.loc[train_test['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()), 'manager_id_unique'] = 1

train_test.loc[train_test['building_id'].isin(buildings_with_one_lot['building_id'].ravel()), 'building_id_unique'] = 1
train_test['building_id_unique'] =  train_test[['building_id','building_id_unique']].apply(lambda x : 1 if x[0] == '0' else x[1], axis=1)
train_test.loc[train_test['display_address'].isin(addresses_with_one_lot['display_address'].ravel()),'display_address_unique'] = 1
categorical = ['building_id_unique', 'manager_id_unique','display_address_unique']

features_to_use.extend(['building_id_unique', 'manager_id_unique','display_address_unique'])


# In[10]:

# replacing ids by counts

categorical = ['building_id', 'manager_id','display_address']
                   
for f in categorical:
    encoder = LabelEncoder()
    encoder.fit(list(train_test[f])) 
    train_test[f] = encoder.transform(train_test[f].ravel())
    
temp = train_test.manager_id.value_counts()
train_test['manager_id_count'] = train_test.manager_id.apply(lambda x: temp[x])

temp = train_test.building_id.value_counts()
train_test['building_id_count'] = train_test.building_id.apply(lambda x: temp[x])

temp = train_test.display_address.value_counts()
train_test['display_address_count'] = train_test.display_address.apply(lambda x: temp[x])


train_test.building_id_count = train_test[['building_id_count', 'building_id_unique']].apply(lambda x : 1 if x[1] == 1 else x[0], axis=1)
train_test.manager_id_count = train_test[['manager_id_count', 'manager_id_unique']].apply(lambda x : 1 if x[1] == 1 else x[0], axis=1)
train_test.display_address_count = train_test[['display_address_count', 'display_address_unique']].apply(lambda x : 1 if x[1] == 1 else x[0], axis=1)

features_to_use.extend(['manager_id_count','building_id_count','display_address_count'])


# In[11]:

features_to_use.extend(['latitude','longitude','listing_id'])


# In[12]:

#median price grouped by managers and neighbourhoods and bedrooms

train_test['median_price_groupby_manager'] = train_test.groupby(['manager_id','bedrooms'])['price'].transform('median') 
features_to_use.append('median_price_groupby_manager')

train_test['median_price_groupby_neighbourhood'] = train_test.groupby(['pred_gm','bedrooms'])['price'].transform('median') - train_test.price
features_to_use.append('median_price_groupby_neighbourhood')


# In[13]:

# extract  features like month, day, hour from date columns and categorizing them, year not taken as every obs is from 2016
train_test["created"] = pd.to_datetime(train_test["created"])

train_test['month'] =train_test["created"].dt.month
train_test['day_of_week'] =train_test["created"].dt.weekday
train_test['day_of_month'] =train_test["created"].dt.day
train_test['hour_of_day'] =train_test["created"].dt.hour

features_to_use.extend(['month','day_of_week','day_of_month','hour_of_day'])


# In[14]:

import random

train_df = train_test.ix[train_df.index]
test_df = train_test.ix[test_df.index]

index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')


# In[15]:

train_test = pd.concat([train_df,test_df])


# In[16]:

# features_to_model = forest_features
features_to_model = features_to_use


# In[17]:

target_num_map = {'high':0, 'medium':1, 'low':2}
y_train = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

X_train = train_test.ix[train_df.index][features_to_model]
X_test = train_test.ix[test_df.index][features_to_model]

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# In[18]:
print 'Features created'

X_train.head()


print 'Fitting model' 

from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

x_train = np.array(X_train)
x_test  = np.array(X_test)  

from sklearn.preprocessing import PolynomialFeatures    
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)    
square = FunctionTransformer(np.square)    

#poly = PolynomialFeatures(degree=2)
x_train[np.isnan(x_train)] = 0
x_test[np.isnan(x_test)] = 0

x_train[np.isinf(x_train)] = 0
x_test[np.isinf(x_test)] = 0

x_train = np.column_stack((x_train,square.transform(x_train),transformer.transform(x_train)))
x_test = np.column_stack((x_test,square.transform(x_test),transformer.transform(x_test)))

x_train[np.isnan(x_train)] = 0
x_test[np.isnan(x_test)] = 0

x_train[np.isinf(x_train)] = 0
x_test[np.isinf(x_test)] = 0

for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

x_train[np.isnan(x_train)] = 0
x_test[np.isnan(x_test)] = 0

x_train[np.isinf(x_train)] = 0
x_test[np.isinf(x_test)] = 0

print x_train.shape
print x_test.shape

train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(x_train.shape[0])) ]
test_stacker=[[0.0 for s in range(3)]   for k in range (0,(x_test.shape[0]))]

X = np.array(X_train)
cv_scores = []
oof_preds = []

from sklearn.linear_model import Perceptron,SGDClassifier 

#model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.001, l1_ratio= 0.3, n_iter= 250, shuffle=True, 
#                       verbose=1 , epsilon=0.1, n_jobs=-1, random_state= 2 , learning_rate='optimal', eta0=0.0, power_t=0.5, 
#                       class_weight=None, warm_start=False, average=False)
model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio= 0.25, n_iter= 2000, shuffle=True, 
                       verbose=0 , epsilon=0.1, n_jobs=-1, random_state= 2 , learning_rate='optimal', eta0=0.0, power_t=0.5, 
                       class_weight=None, warm_start=False, average=False)

# StratifiedKFold
# kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=2016)
# for dev_index, val_index in kf.split(range(x_train.shape[0]),y_train):

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(x_train.shape[0])):
        dev_X, val_X = x_train[dev_index,:], x_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        model.fit(dev_X, dev_y)
        preds =  model.predict_proba(val_X)
        cv_scores.append(log_loss(val_y, preds))
        predictions = preds.reshape( val_X.shape[0], 3)
        print(cv_scores)
        no=0
        for real_index in val_index:
            for d in range (0,3):
                train_stacker[real_index][d]=(predictions[no][d])
            no+=1

np.savetxt("stnet/train_stacker_sgd_tf.csv", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

model.fit(x_train,y_train)

pred_nn = model.predict_proba(x_test)

np.savetxt("stnet/test_stacker_sgd_tf.csv", pred_nn, delimiter=",", fmt='%.6f')

#sub = pd.DataFrame(data = {'listing_id': test_df['listing_id'].ravel()})
#sub['low']  = pred_nn[:, 2]
#sub['medium'] = pred_nn[:, 1]
#sub['high'] = pred_nn[:, 0]
#sub.to_csv("model_nn_16.csv", index = False, header = True)

#print(results.mean())
