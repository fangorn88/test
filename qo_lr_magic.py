from __future__ import  division
import numpy as np
import pandas as pd
import timeit


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

df = pd.read_csv('train.csv')
y = df['is_duplicate'].values
                 
x_train = np.loadtxt("train_nlp_feat.gz", delimiter=",")
x_test = np.loadtxt("test_nlp_feat.gz", delimiter=",")

print x_train.shape

print x_test.shape

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0

for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

x_train[~np.isfinite(x_train)] = 0
x_test[~np.isfinite(x_test)] = 0


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

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                            intercept_scaling=1, class_weight=None, random_state=None, solver='sag', 
                            max_iter=500, verbose=1, warm_start=False, n_jobs=12)
 
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

np.savetxt("train_stacker_lr3.csv", train_stacker, delimiter=",", fmt='%.6f')

#print 'finished out of fold predictions on training set'             

np.savetxt("test_stacker_lr3.csv", np.array(b['sum']), delimiter=",", fmt='%.6f')

#sub = pd.DataFrame()
#sub['test_id'] = x4t['test_id']
#sub['is_duplicate'] = b['sum']
#sub.to_csv("xgb2016_5f_0.167x_0.01lr.csv", index=False)
