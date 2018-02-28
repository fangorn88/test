

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



target_num_map = {'high':0, 'medium':1, 'low':2}
y_train = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

x_train = np.loadtxt("stnet/train_15.csv", delimiter=",")
x_test = np.loadtxt("stnet/test_15.csv", delimiter=",")

print x_train.shape
print x_test.shape

print 'Fitting model' 

from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(x_train.shape[0])) ]
test_stacker=[[0.0 for s in range(3)]   for k in range (0,(x_test.shape[0]))]


from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier 
from sklearn.linear_model import Perceptron,SGDClassifier,RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC


model = ExtraTreesClassifier(n_estimators=500, criterion='entropy', max_depth= 13, max_features='auto', 
                                               max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, 
                                               oob_score=False, n_jobs=-1, random_state=None, verbose=1 , warm_start=False, 
                                               class_weight=None)



cv_scores = []
oof_preds = []

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

np.savetxt("stnet/train_stacker_et_meta_md13n.csv", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

model.fit(x_train,y_train)

pred_nn = model.predict_proba(x_test)

np.savetxt("stnet/test_stacker_et_meta_md13n.csv", pred_nn, delimiter=",", fmt='%.6f')

