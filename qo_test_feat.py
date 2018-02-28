

import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')

dtest = pd.read_csv('test.csv', sep=',')
dtest = dtest.drop(['test_id'], axis=1)


dtest['len_q1'] = dtest.question1.apply(lambda x: len(str(x)))
dtest['len_q2'] = dtest.question2.apply(lambda x: len(str(x)))
dtest['diff_len'] = dtest.len_q1 - dtest.len_q2
dtest['len_char_q1'] = dtest.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
dtest['len_char_q2'] = dtest.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
dtest['len_word_q1'] = dtest.question1.apply(lambda x: len(str(x).split()))
dtest['len_word_q2'] = dtest.question2.apply(lambda x: len(str(x).split()))
dtest['common_words'] = dtest.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
dtest['fuzz_qratio'] = dtest.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_WRatio'] = dtest.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_partial_ratio'] = dtest.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_partial_token_set_ratio'] = dtest.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_partial_token_sort_ratio'] = dtest.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_token_set_ratio'] = dtest.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
dtest['fuzz_token_sort_ratio'] = dtest.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

print ' basic features created'

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
dtest['wmd'] = dtest.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
norm_model.init_sims(replace=True)
dtest['norm_wmd'] = dtest.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((dtest.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(dtest.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((dtest.shape[0], 300))
for i, q in tqdm(enumerate(dtest.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

dtest['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

dtest['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
dtest['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
dtest['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
dtest['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

cPickle.dump(question1_vectors, open('q1_w2v_test.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('q2_w2v_test.pkl', 'wb'), -1)

dtest.to_csv('quora_features_test.csv', index=False)

print 'All features created'