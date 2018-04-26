import re
import gc
import string

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.05,
              'application': 'regression',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.8,
              'nthread': 3,
              'lambda_l1': 1,
              'lambda_l2': 1}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=2000,
                      valid_sets=watchlist,
                      verbose_eval=100)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()

#if not is_in_cache('cleaned'):
print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~')
print_step('Merging')
merge = pd.concat([train, test])

print('~~~~~~~~~~~~~~~')
print_step('Impute 1/6')
merge['param_1_missing'] = merge['param_1'].isna().astype(int)
merge['param_1'].fillna('missing', inplace=True)
print_step('Impute 2/6')
merge['param_2_missing'] = merge['param_2'].isna().astype(int)
merge['param_2'].fillna('missing', inplace=True)
print_step('Impute 3/6')
merge['param_3_missing'] = merge['param_3'].isna().astype(int)
merge['param_3'].fillna('missing', inplace=True)
print_step('Impute 4/6')
merge['price_missing'] = merge['price'].isna().astype(int)
merge['price'].fillna(merge['price'].median(), inplace=True)
merge['has_price'] = (merge['price'] > 0).astype(int)
print_step('Impute 5/6')
merge['description_missing'] = merge['description'].isna().astype(int)
merge['description'].fillna('', inplace=True)
print_step('Impute 6/6')
merge['image_missing'] = merge['image'].isna().astype(int)
merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')

print('~~~~~~~~~~~~~~')
print_step('Basic NLP 1/20')
merge['num_words_description'] = merge['description'].apply(lambda x: len(str(x).split()))
print_step('Basic NLP 2/20')
merge['num_words_title'] = merge['title'].apply(lambda x: len(str(x).split()))
print_step('Basic NLP 3/20')
merge['num_chars_description'] = merge['description'].apply(lambda x: len(str(x)))
print_step('Basic NLP 4/20')
merge['num_chars_title'] = merge['title'].apply(lambda x: len(str(x)))
print_step('Basic NLP 5/20')
merge['num_capital_description'] = merge['description'].apply(lambda x: len([c for c in x if c.isupper()]))
print_step('Basic NLP 6/20')
merge['num_capital_title'] = merge['title'].apply(lambda x: len([c for c in x if c.isupper()]))
print_step('Basic NLP 7/20')
merge['num_lowercase_description'] = merge['description'].apply(lambda x: len([c for c in x if c.islower()]))
print_step('Basic NLP 8/20')
merge['num_lowercase_title'] = merge['title'].apply(lambda x: len([c for c in x if c.islower()]))
print_step('Basic NLP 9/20')
merge['capital_per_char_description'] = merge['num_capital_description'] / merge['num_chars_description']
merge['capital_per_char_description'].fillna(0, inplace=True)
print_step('Basic NLP 10/20')
merge['capital_per_char_title'] = merge['num_capital_title'] / merge['num_chars_title']
merge['capital_per_char_title'].fillna(0, inplace=True)
print_step('Basic NLP 11/20')
merge['num_punctuations'] = merge['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
print_step('Basic NLP 12/20')
merge['punctuation_per_char'] = merge['num_punctuations'] / merge['num_chars_description']
merge['punctuation_per_char'].fillna(0, inplace=True)
print_step('Basic NLP 13/20')
merge['num_words_upper_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print_step('Basic NLP 14/20')
merge['num_words_upper_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print_step('Basic NLP 15/20')
merge['num_words_lower_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
print_step('Basic NLP 16/20')
merge['num_words_lower_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
print_step('Basic NLP 17/20')
merge['num_words_entitled_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
print_step('Basic NLP 18/20')
merge['num_words_entitled_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
print_step('Basic NLP 19/20')
merge['chars_per_word_description'] = merge['num_chars_description'] / merge['num_words_description']
merge['chars_per_word_description'].fillna(0, inplace=True)
print_step('Basic NLP 20/20')
merge['chars_per_word_title'] = merge['num_chars_title'] / merge['num_words_title']
merge['chars_per_word_title'].fillna(0, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~')
print_step('Activation Date')
merge['activation_date'] = pd.to_datetime(merge['activation_date'])
merge['day_of_week'] = merge['activation_date'].dt.weekday
merge['weekend'] = ((merge['day_of_week'] == 5) | (merge['day_of_week'] == 6)).astype(int)

print('~~~~~~~~~~~~~')
print_step('Dropping')
drops = ['activation_date', 'title', 'description']
merge.drop(drops, axis=1, inplace=True)
currently_unused = ['region', 'user_id', 'city', 'image', 'item_seq_number']
merge.drop(currently_unused, axis=1, inplace=True)

print('~~~~~~~~~~~~')
print_step('Dummies 1/2')
print(merge.shape)
dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'param_1',
              'param_2', 'param_3', 'image_top_1', 'day_of_week']
for col in dummy_cols:
    le = LabelEncoder()
    merge[col] = le.fit_transform(merge[col])
print_step('Dummies 2/2')
ohe = OneHotEncoder(categorical_features=[merge.columns.get_loc(c) for c in dummy_cols])
merge = ohe.fit_transform(merge)
print(merge.shape)

print_step('Unmerge')
merge = merge.tocsr()
dim = train.shape[0]
train_ = merge[:dim]
test_ = merge[dim:]


print('~~~~~~~~~~~~')
print_step('Run LGB')
print(train_.shape)
print(test_.shape)

results = run_cv_model(train_, test_, target, runLGB, rmse, 'lgb')
import pdb
pdb.set_trace()

#print('~~~~~~~~~~')
#print_step('Cache')
#save_in_cache('lvl1_lgb', train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb.csv', index=False)
print_step('Done!')

# TODO
# OHE region and city
# Include item_seq_number as a straightforward numeric
# Basic NLP
    # merge['sentence'] = merge['description'].apply(lambda x: [s for s in re.split(r'[.!?\n]+', str(x))])
    # merge['num_sentence'] = merge['sentence'].apply(lambda x: len(x))
    # merge['sentence_mean'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.mean(x))
    # merge['sentence_max'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: max(x) if len(x) > 0 else 0)
    # merge['sentence_min'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: min(x) if len(x) > 0 else 0)
    # merge['sentence_std'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.std(x))
    # merge['words_per_sentence'] = merge['num_words'] / merge['num_sentence']
    # merge.drop('sentence', axis=1, inplace=True)
    # description_words_per_title_words
    # description_characters_per_title_characters
    # Number english characters
    # Number russian characters
    # Number english words
    # Number russian words
# Frequency encode user_id, city, region, parent_category_name, category_name, param_1, param_2, param_3, image_top_1, day_of_week
# encode(mean, median, std, min, max) of target, price, item_seq_number with user_id
# encode(mean, median, std, min, max) of target, price with city, region, parent_category_name, category_name, param_1, param_2, param_3, image_top_1, user_type, day_of_week
# TFIDF concat(title, description) -> LR, SelectKBest, SVD, Embedding
    # https://www.kaggle.com/iggisv9t/basic-tfidf-on-text-features-0-233-lb
# TFIDF concat(title, description, param_1, param_2, param_3, parent_category_name, category_name) -> LR, SelectKBest, SVD, Embedding
# Check feature impact in DR
# Tune models some
# Train classification and regression
# Image analysis
    # img_hash.py?
    # https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16)
    # Contrast? https://dsp.stackexchange.com/questions/3309/measuring-the-contrast-of-an-image
    # https://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/
    # NNs?
# Train more models (Ridge, FM, Ridge, NNs)
# Look to DonorsChoose
    # https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x/code
    # https://www.kaggle.com/fizzbuzz/the-all-in-one-model
    # https://www.kaggle.com/emotionevil/beginners-workflow-meanencoding-lgb-nn-ensemble
    # https://www.kaggle.com/safavieh/ultimate-feature-engineering-xgb-lgb-nn
    # https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
    # https://www.kaggle.com/nicapotato/abc-s-of-tf-idf-boosting-0-798
# https://www.kaggle.com/c/avito-duplicate-ads-detection
# Use train_active and test_active somehow?
