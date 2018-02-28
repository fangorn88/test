

import numpy as np
import pandas as pd
import timeit
import os
import gc


x1 = pd.read_csv('quora_features.csv')
x2 = pd.read_csv('magicfeat_train.csv')
x3 = pd.read_csv('fanokas.csv')
x4 = pd.read_csv('owl_feat.csv')

col_to_drop = ['z_len1','z_len2','z_word_len1','z_word_len2','z_word_match']

x4.drop(col_to_drop, inplace = True, axis = 1)


xf = pd.concat([x1,x2.ix[:,2:],x3.ix[:,1:],x4.ix[:,9:]], axis = 1)

X_train = xf.ix[:,3:]
X_train.drop('is_duplicate', inplace = True, axis = 1)
y_train = xf['is_duplicate'].values


x1t = pd.read_csv('test_features.csv')
x2t = pd.read_csv('magicfeat_test.csv')
x3t = pd.read_csv('fanokas_test.csv')
x4t = pd.read_csv('owl_feat_test.csv')

col_to_drop = ['z_len1','z_len2','z_word_len1','z_word_len2','z_word_match']

x4t.drop(col_to_drop, inplace = True, axis = 1)


xft = pd.concat([x1t,x2t.ix[:,2:],x3t.ix[:,1:],x4t.ix[:,6:]], axis = 1)

X_test = xft.ix[:,2:]

print 'training and test features created' 

pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = float(((float(len(pos_train)) / (float(len(pos_train)) + float(len(neg_train)))) / p) - 1)

print scale
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print float((float(len(pos_train)) / (float(len(pos_train)) + float(len(neg_train)))))

X_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

print 'Imbalancing handled' 

#from sklearn.cross_validation import train_test_split
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

print 'Normalizing training and test data'

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)


X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

x_train = np.array(X_train)
x_test =  np.array(X_test)

print x_train.shape
print x_test.shape

for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])

    
import os

os.environ['KERAS_BACKEND']='theano'

from keras.layers import Dense,Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# import numpy
# import numpy
# import pandas
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

#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)

# c, r = y.shape
# y = y.reshape(c,)

 

# Function to create model, required for KerasClassifier
def create_model():
# create model
    model = Sequential()
    
    model.add(Dense(44, input_dim=x_train.shape[1], kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

#    model.add(Dense(136, kernel_initializer='he_normal', activation='relu')) 
#    model.add(BatchNormalization())
#    model.add(Dropout(0.6))

    model.add(Dense(24, kernel_initializer='he_normal', activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#     model.add(Dense(100, input_dim = 68, init = 'he_normal'))
#     model.add(PReLU())
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
        
# #     model.add(Dense(200, init = 'he_normal'))
# #     model.add(PReLU())
# #     model.add(BatchNormalization())    
# #     model.add(Dropout(0.2))
    
#     model.add(Dense(50, init = 'he_normal'))
#     model.add(PReLU())
#     model.add(BatchNormalization())    
#     model.add(Dropout(0.2))
    
#     model.add(Dense(3, init='he_normal', activation='softmax'))
#     adam = optimizers.Adam(lr=1.0)
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return(model)

# y_arr = pd.DataFrame.as_matrix(pd.DataFrame(y)).ravel()
# y_arr = np.array(y)
# fix random seed for reproducibility

seed = 7

earlyStopping=callbacks.EarlyStopping(monitor='loss', patience = 1, verbose=0, mode='auto')

model = KerasClassifier(build_fn=create_model, epochs=40, batch_size=50, verbose=1)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = model_selection.cross_val_score(model, x_train,y_train, cv=kfold, scoring = 'neg_log_loss',fit_params={'callbacks': [earlyStopping]})
print(results.mean())


model.fit(x_train,y_train)

pred_nn = model.predict_proba(x_test)


sub = pd.DataFrame()
sub['test_id'] = x4t['test_id']
sub['is_duplicate'] = pred_nn
#sub.to_csv('simple_xgb_ab_mg_an.csv', index=False)# sub['low']  = np.clip(pred_nn[:, 0], clip, 1-clip)
# sub['medium'] = np.clip(pred_nn[:, 1], clip, 1-clip)
# sub['high'] = np.clip(pred_nn[:, 2], clip, 1-clip)
sub.to_csv("submission_nn_epoch_40_bs_50.csv", index = False, header = True)
