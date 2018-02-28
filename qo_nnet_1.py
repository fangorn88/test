
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

import os
os.environ['KERAS_BACKEND']='theano'

from keras.layers import Dense,Dropout
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras import regularizers
from keras import callbacks

#from keras.utils.np_utils import to_categorical

#categorical_labels = to_categorical(int_labels, num_classes=None)

# Function to create model, required for KerasClassifier
def create_model():
# create model
    model = Sequential()

    model.add(Dense(211, input_dim = x_train.shape[1], kernel_initializer = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
        
#    model.add(Dense(128, kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(.0001)))
#    model.add(PReLU())
#    model.add(BatchNormalization())    
#    model.add(Dropout(0.5))
    
    model.add(Dense(64, kernel_initializer = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.5))
    
    model.add(Dense(1, init='he_normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 2016

earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience = 1, verbose=1, mode='auto')

model = KerasClassifier(build_fn=create_model, epochs=40, batch_size=1024, verbose=1)


from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# x_train = np.array(X)
#y_train = np.array(y)


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

        model.fit(dev_X,dev_y,validation_data=(val_X, val_y), callbacks = [earlyStopping])
        
        preds = model.predict(val_X)

        cv_scores.append(log_loss(val_y, preds))
        
        preds_tr = model.predict(x_test)

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



