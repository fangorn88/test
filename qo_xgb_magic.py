from __future__ import  division
import numpy as np
import pandas as pd
import timeit


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

x4 = pd.read_csv('owl_feat.csv')

#col_to_drop = ['z_len1','z_len2','z_word_len1','z_word_len2','z_word_match']

#x4.drop(col_to_drop, inplace = True, axis = 1)

xf = x4.ix[:,9:]

x4t = pd.read_csv('owl_feat_test.csv')

xft = x4t.ix[:,6:]

X = xf

y = x4['is_duplicate'].values

#X.drop('is_duplicate', inplace = True, axis = 1)

x_train = np.array(X)

print x_train.shape

xtest = xft

x_test = np.array(xtest)

print x_test.shape

RS = 2016
ROUNDS = 400

print("Started")
np.random.seed(RS)
input_folder = ''

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 6
params['seed'] = RS
params['gamma'] = 0.5
params['subsample'] = 0.75
params['colsample_bytree'] = 0.75
params['min_child_weight'] = 10
params['reg_alpha'] = 2
params['n_jobs'] = -1

from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

y_train = np.array(y)

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
        p = 0.1742
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = np.concatenate((neg_train, neg_train))
            scale -=1
        neg_train = np.concatenate((neg_train, neg_train[:int(scale * len(neg_train))]))
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        Xd = np.concatenate((pos_train, neg_train))
        yd = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train  

        pos_train = val_X[val_y == 1]
        neg_train = val_X[val_y == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
        p = 0.1742
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = np.concatenate((neg_train, neg_train))
            scale -=1
        neg_train = np.concatenate((neg_train, neg_train[:int(scale * len(neg_train))]))
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        Xv = np.concatenate((pos_train, neg_train))
        yv = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train  

        print Xd.shape
        print Xv.shape

        d_train = xgb.DMatrix(Xd, label=yd)
        d_valid = xgb.DMatrix(Xv, label=yv)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        bst = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=100)

        preds = bst.predict(d_valid)

        cv_scores.append(log_loss(yv, preds))
        
        d_test = xgb.DMatrix(x_test)
        preds_tr = bst.predict(d_test)

        a = np.column_stack((a,preds_tr))
        print(cv_scores)

        d_valorg = xgb.DMatrix(val_X, label=val_y)
        predsorg = bst.predict(d_valorg)

#         predictions = preds.reshape(-1,1)
        no=0
        for real_index in val_index:
            for d in range (0,1):
                train_stacker[real_index][d]=(predsorg[no])
            no+=1

b = pd.DataFrame(a)

b['sum'] = b.sum(axis = 1)/5

np.savetxt("train_stacker_xgbowl.csv", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

np.savetxt("test_stacker_xgbowl.csv", np.array(b['sum']), delimiter=",", fmt='%.6f')

