
from __future__ import  division
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

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print 'Files imported'

#creating test_train and y_train
y = train_df['is_duplicate'].values

y_train = np.array(y)


print 'Creating features' 

x_train = np.loadtxt("x_train.gz", delimiter=",")
x_test = np.loadtxt("x_test.gz", delimiter=",")

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0

for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0
    
print x_train.shape
print x_test.shape

print 'Fitting model' 

from sklearn import model_selection
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
# x_train = np.array(X)
#y_train = np.array(y)

model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=12,max_features= 0.3, min_samples_split =10                                               ,bootstrap=True,n_jobs=-1, oob_score = True, random_state=2016, verbose=1)

train_stacker=[ [0.0 for s in range(1)]  for k in range (0,(x_train.shape[0])) ]

cv_scores = []
oof_preds = []
a = [0 for x in range(2345796)]


# StratifiedKFold
# kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
# for dev_index, val_index in kf.split(range(x_train.shape[0]),y_train):
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(x_train.shape[0])):
        dev_X, val_X = x_train[dev_index,:], x_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        print dev_X.shape
        print val_X.shape
        
        pos_train = dev_X[dev_y == 1]
        neg_train = dev_X[dev_y == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
        p = 0.165
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = np.concatenate((neg_train, neg_train))
            scale -=1
        neg_train = np.concatenate((neg_train, neg_train[:int(scale * len(neg_train))]))
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        dev_X = np.concatenate((pos_train, neg_train))
        dev_y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train  

        pos_train = val_X[val_y == 1]
        neg_train = val_X[val_y == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
        p = 0.165
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = np.concatenate((neg_train, neg_train))
            scale -=1
        neg_train = np.concatenate((neg_train, neg_train[:int(scale * len(neg_train))]))
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        val_X = np.concatenate((pos_train, neg_train))
        val_y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train  

        print dev_X.shape
        print val_X.shape

        model.fit(dev_X,dev_y)
        
        preds = model.predict_proba(val_X)

        cv_scores.append(log_loss(val_y, preds))
        
        preds_tr = model.predict_proba(x_test)

        a = np.column_stack((a,preds_tr))
#         cv_scores.append(log_loss(val_y, preds*0.99))
#         cv_scores.append(log_loss(val_y, preds*0.98))
#         cv_scores.append(log_loss(val_y, preds*0.95))
#         cv_scores.append(log_loss(val_y, preds*0.90))

        print(cv_scores)
#         break

        no=0
        for real_index in val_index:
#             for d in range (0,1):
            train_stacker[real_index]= preds
            no+=1


b = pd.DataFrame(a)

b['sum'] = b.sum(axis = 1)/5
train_stacker = np.array(train_stacker)

print train_stacker.shape

np.savetxt('oof_rf1_train.gz', np.array(train_stacker), delimiter=",", fmt='%.5f')
np.savetxt('oof_rf1_test.gz', np.array(b['sum']), delimiter=",", fmt='%.5f')
