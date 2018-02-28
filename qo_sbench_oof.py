import time
import pandas as pd
import numpy as np
import networkx as nx

from sklearn import model_selection
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

trainDF = pd.read_csv('train.csv')

trainDF.ix[trainDF['question1'].isnull(),['question1','question2']] = 'random empty question'
trainDF.ix[trainDF['question2'].isnull(),['question1','question2']] = 'random empty question'

featureExtractionStartTime = time.time()

maxNumFeatures = 300000

BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(1,10), 
                                      binary=True, lowercase=True)
BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question1'])
trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question2'])
lables = np.array(trainDF.ix[:,'is_duplicate'])

featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0
print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))

X = (trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
#X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int) + \
#      trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep)
y = lables

testDF = pd.read_csv('test.csv')
testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'
testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'

testQuestion1_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:,'question1'])
testQuestion2_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:,'question2'])

X_test = (testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)

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
params['eta'] = 0.3
params['max_depth'] = 30
params['seed'] = RS
params['gamma'] = 2
params['subsample'] = 0.75
params['colsample_bytree'] = 0.75
params['min_child_weight'] = 10
params['reg_alpha'] = 2
#params['reg_lambda'] = 2
params['n_jobs'] = 16

from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

y_train = np.array(y)

train_stacker=[ [0.0 for s in range(1)]  for k in range (0,(X.shape[0])) ]

cv_scores = []
oof_preds = []
a = [0 for x in range(2345796)]
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(X.shape[0])):
        dev_X, val_X = X[dev_index,:], X[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        print dev_X.shape
        print val_X.shape

        d_train = xgb.DMatrix(dev_X, label=dev_y)
        d_valid = xgb.DMatrix(val_X, label=val_y)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        bst = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=25, verbose_eval=100)
        # ntree_limit=model.best_ntree_limit
        m = 0.1742 / 0.369 
        n = (1 - 0.1742) / (1 - 0.369)
        
        pds = bst.predict(d_valid, ntree_limit=bst.best_ntree_limit)

        preds = m * pds / (m * pds + n * (1 - pds))
        cv_scores.append(log_loss(val_y, preds))

        print(cv_scores)
#         break
        
        d_test = xgb.DMatrix(X_test)
        ptr = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)
        

        preds_tr = m * ptr / (m * ptr + n * (1 - ptr))

        a = np.column_stack((a,preds_tr))

        no=0
        for real_index in val_index:
            for d in range (0,1):
                train_stacker[real_index][d]=(preds[no])
            no+=1

b = pd.DataFrame(a)

b['sum'] = b.sum(axis = 1)/5

np.savetxt("train_stacker_sbxgb1.gz", train_stacker, delimiter=",", fmt='%.6f')

print 'finished out of fold predictions on training set'             

np.savetxt("test_stacker_sbxgb1.gz", np.array(b['sum']), delimiter=",", fmt='%.6f')
            