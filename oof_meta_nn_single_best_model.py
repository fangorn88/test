

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


from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier 
from sklearn.linear_model import Perceptron,SGDClassifier,RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC


print 'Fitting model' 

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
# from keras.models import Sequential
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

# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
 

# Function to create model, required for KerasClassifier
def create_model():
# create model
    model = Sequential()

#    model.add(Dense(512, input_dim = x_train.shape[1], kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(.0001)))
    model.add(Dense(45, input_dim = x_train.shape[1], kernel_initializer = 'he_normal',activation='relu'))
#    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
#    model.add(Dense(32, kernel_initializer = 'he_normal',activation='relu'))
#    model.add(PReLU())
#    model.add(BatchNormalization())    
#    model.add(Dropout(0.5))
    
    model.add(Dense(15, kernel_initializer = 'he_normal',activation='relu'))
#    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(3, init='he_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 45

earlyStopping=callbacks.EarlyStopping(monitor='loss', patience = 1, verbose=0, mode='auto')

model = KerasClassifier(build_fn=create_model, epochs=40, batch_size=50, verbose=1)

# evaluate using 10-fold cross validation
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = model_selection.cross_val_score(model, x_train,y_train, cv=kfold, scoring = 'neg_log_loss',fit_params={'callbacks': [earlyStopping]})


from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

print x_train.shape
print x_test.shape

train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(x_train.shape[0])) ]
test_stacker=[[0.0 for s in range(3)]   for k in range (0,(x_test.shape[0]))]

cv_scores = []
oof_preds = []

# StratifiedKFold
# kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=2016)
# for dev_index, val_index in kf.split(range(x_train.shape[0]),y_train):

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(x_train.shape[0])):
        dev_X, val_X = x_train[dev_index,:], x_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        model.fit(dev_X, dev_y,callbacks = [earlyStopping])
        preds =  model.predict_proba(val_X)
        cv_scores.append(log_loss(val_y, preds))
        predictions = preds.reshape( val_X.shape[0], 3)
        print(cv_scores)
        no=0
        for real_index in val_index:
            for d in range (0,3):
                train_stacker[real_index][d]=(predictions[no][d])
            no+=1

np.savetxt("stnet/train_stacker_meta_nn6.csv", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

model.fit(x_train,y_train,callbacks = [earlyStopping])

pred_nn = model.predict_proba(x_test)

np.savetxt("stnet/test_stacker_meta_nn6.csv", pred_nn, delimiter=",", fmt='%.6f')

