from __future__ import  division
import numpy as np
import pandas as pd
import timeit


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

x1 = pd.read_csv('train_features.csv')
x2 = pd.read_csv('magicfeat_train.csv')
x3 = pd.read_csv('fanokas.csv')
x4 = pd.read_csv('owl_feat.csv')
x5 = pd.read_csv('jacfeat.csv')

col_to_drop = ['z_len1','z_len2','z_word_len1','z_word_len2','z_word_match']

x4.drop(col_to_drop, inplace = True, axis = 1)

xf = pd.concat([x1,x2.ix[:,2:],x3.ix[:,1:],x4.ix[:,9:],x5.ix[:,1:]], axis = 1)

x1t = pd.read_csv('test_features.csv')
x2t = pd.read_csv('magicfeat_test.csv')
x3t = pd.read_csv('fanokas_test.csv')
x4t = pd.read_csv('owl_feat_test.csv')
x5t = pd.read_csv('jacfeat_test.csv')

col_to_drop = ['z_len1','z_len2','z_word_len1','z_word_len2','z_word_match']

x4t.drop(col_to_drop, inplace = True, axis = 1)

xft = pd.concat([x1t,x2t.ix[:,2:],x3t.ix[:,1:],x4t.ix[:,6:],x5t.ix[:,1:]], axis = 1)

X = xf.ix[:,2:]

y = X['is_duplicate'].values

X.drop('is_duplicate', inplace = True, axis = 1)

col_to_drop = ['q1_hash','q2_hash']

X.drop(col_to_drop, inplace = True, axis = 1)

sbtr = np.loadtxt("sbenchfeat_tsvd100_train.gz", delimiter=",")
rstr = np.loadtxt("russ_tr.gz", delimiter=",")
rspacetr = np.loadtxt("russpacy_tr.gz", delimiter=",")

sbts = np.loadtxt("sbenchfeat_tsvd100_test.gz", delimiter=",")
rsts = np.loadtxt("russ_ts.gz", delimiter=",")
rspacets = np.loadtxt("russpacy_ts.gz", delimiter=",")


magic2tr = np.loadtxt("magic2_tr.gz", delimiter=",")
magic2ts = np.loadtxt("magic2_ts.gz", delimiter=",")

magic3tr = np.loadtxt("magic3_tr.gz", delimiter=",")
magic3ts = np.loadtxt("magic3_ts.gz", delimiter=",")

x_train = np.column_stack((np.array(X),rstr[:,0:17],rstr[:,18:20],rspacetr[:,0:3],sbtr,magic2tr,magic3tr))

print x_train.shape

xtest = xft.ix[:,2:]

col_to_drop = ['q1_hash','q2_hash']
xtest.drop(col_to_drop, inplace = True, axis = 1)

x_test = np.column_stack((np.array(xtest),rsts[:,0:17],rsts[:,18:20],rspacets[:,0:3],sbts,magic2ts,magic3ts))

print x_test.shape

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0

for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0


RS = 2016
ROUNDS = 400

print("Started")
np.random.seed(RS)
input_folder = ''

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.linear_model import Perceptron,SGDClassifier 

model = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio= 0.25, n_iter= 2000, shuffle=True, 
                       verbose=0 , epsilon=0.1, n_jobs=-1, random_state= 2 , learning_rate='optimal', eta0=0.0, power_t=0.5, 
                       class_weight=None, warm_start=False, average=False)

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
        p = 0.165
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
        p = 0.165
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

        model.fit(Xd, yd) 

        preds = model.predict_proba(Xv)

        cv_scores.append(log_loss(yv, preds))
        
        preds_tr = model.predict_proba(x_test)

        a = np.column_stack((a,preds_tr[:,1]))
        print(cv_scores)

        predsorg = model.predict_proba(val_X)

#         predictions = preds.reshape(-1,1)
        no=0
        for real_index in val_index:
            for d in range (0,1):
                train_stacker[real_index][d]=(predsorg[no][1])
            no+=1

b = pd.DataFrame(a)

b['sum'] = b.sum(axis = 1)/5

np.savetxt("train_stacker_sgd1.csv", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

np.savetxt("test_stacker_sgd1.csv", np.array(b['sum']), delimiter=",", fmt='%.6f')

#sub = pd.DataFrame()
#sub['test_id'] = x4t['test_id']
#sub['is_duplicate'] = b['sum']
#sub.to_csv("xgb2016_5f_0.167x_0.01lr.csv", index=False)
