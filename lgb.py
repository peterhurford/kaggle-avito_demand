import string

import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, TargetEncoder
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

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)


if not is_in_cache('data_with_fe'):
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
    merge['image_top_1'] = train['image_top_1'].fillna(-8).astype(str)


    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Basic NLP 1/32')
    merge['num_words_description'] = merge['description'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 2/32')
    merge['num_words_title'] = merge['title'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 3/32')
    merge['num_chars_description'] = merge['description'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 4/32')
    merge['num_chars_title'] = merge['title'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 5/32')
    merge['num_capital_description'] = merge['description'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 6/32')
    merge['num_capital_title'] = merge['title'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 7/32')
    merge['num_lowercase_description'] = merge['description'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 8/32')
    merge['num_lowercase_title'] = merge['title'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 9/32')
    merge['capital_per_char_description'] = merge['num_capital_description'] / merge['num_chars_description']
    merge['capital_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 10/32')
    merge['capital_per_char_title'] = merge['num_capital_title'] / merge['num_chars_title']
    merge['capital_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 11/32')
    merge['num_punctuations'] = merge['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    print_step('Basic NLP 12/32')
    merge['punctuation_per_char'] = merge['num_punctuations'] / merge['num_chars_description']
    merge['punctuation_per_char'].fillna(0, inplace=True)
    print_step('Basic NLP 13/32')
    merge['num_words_upper_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print_step('Basic NLP 14/32')
    merge['num_words_upper_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print_step('Basic NLP 15/32')
    merge['num_words_lower_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    print_step('Basic NLP 16/32')
    merge['num_words_lower_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    print_step('Basic NLP 17/32')
    merge['num_words_entitled_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print_step('Basic NLP 18/32')
    merge['num_words_entitled_title'] = merge['title'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print_step('Basic NLP 19/32')
    merge['chars_per_word_description'] = merge['num_chars_description'] / merge['num_words_description']
    merge['chars_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 20/32')
    merge['chars_per_word_title'] = merge['num_chars_title'] / merge['num_words_title']
    merge['chars_per_word_title'].fillna(0, inplace=True)
    print_step('Basic NLP 21/32')
    merge['description_words_per_title_words'] = merge['num_words_description'] / merge['num_words_title']
    print_step('Basic NLP 22/32')
    merge['description_words_per_title_words'].fillna(0, inplace=True)
    print_step('Basic NLP 23/32')
    merge['description_chars_per_title_chars'] = merge['num_chars_description'] / merge['num_chars_title']
    print_step('Basic NLP 24/32')
    merge['description_chars_per_title_chars'].fillna(0, inplace=True)
    print_step('Basic NLP 25/32')
    merge['num_english_chars_description'] = merge['description'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 26/32')
    merge['num_english_chars_title'] = merge['title'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 27/32')
    merge['english_chars_per_char_description'] = merge['num_english_chars_description'] / merge['num_chars_description']
    merge['english_chars_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 28/32')
    merge['english_chars_per_char_title'] = merge['num_english_chars_title'] / merge['num_chars_title']
    merge['english_chars_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 29/32')
    merge['num_english_words_description'] = merge['description'].apply(lambda ss: len([w for w in ss.lower().translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').split(' ') if all([s in string.ascii_lowercase for s in w]) and len(w) > 0]))
    print_step('Basic NLP 30/32')
    merge['num_english_words_title'] = merge['title'].apply(lambda ss: len([w for w in ss.lower().translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').split(' ') if all([s in string.ascii_lowercase for s in w]) and len(w) > 0]))
    print_step('Basic NLP 31/32')
    merge['english_words_per_char_description'] = merge['num_english_words_description'] / merge['num_words_description']
    merge['english_words_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 32/32')
    merge['english_words_per_char_title'] = merge['num_english_words_title'] / merge['num_words_title']
    merge['english_words_per_char_title'].fillna(0, inplace=True)

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday
    merge['weekend'] = ((merge['day_of_week'] == 5) | (merge['day_of_week'] == 6)).astype(int)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Frequency Encode 1/10')
    merge['user_id_count'] = merge.groupby('user_id')['user_id'].transform('count')
    print_step('Frequency Encode 2/10')
    merge['city_count'] = merge.groupby('city')['city'].transform('count')
    print_step('Frequency Encode 3/10')
    merge['region_count'] = merge.groupby('region')['region'].transform('count')
    print_step('Frequency Encode 4/10')
    merge['parent_category_name_count'] = merge.groupby('parent_category_name')['parent_category_name'].transform('count')
    print_step('Frequency Encode 5/10')
    merge['category_name_count'] = merge.groupby('category_name')['category_name'].transform('count')
    print_step('Frequency Encode 6/10')
    merge['param_1_count'] = merge.groupby('param_1')['param_1'].transform('count')
    print_step('Frequency Encode 7/10')
    merge['param_2_count'] = merge.groupby('param_2')['param_2'].transform('count')
    print_step('Frequency Encode 8/10')
    merge['param_3_count'] = merge.groupby('param_3')['param_3'].transform('count')
    print_step('Frequency Encode 9/10')
    merge['image_top_1_count'] = merge.groupby('image_top_1')['image_top_1'].transform('count')
    print_step('Frequency Encode 10/10')
    merge['day_of_week_count'] = merge.groupby('day_of_week')['day_of_week'].transform('count')

    print('~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Additional FE 1/2')
    merge['num_missing'] = merge.isna().sum(axis=1)
    print_step('Additional FE 2/2')
    merge['adjusted_item_seq_number'] = merge['item_seq_number'] - merge.groupby('user_id')['item_seq_number'].transform('min')

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    drops = ['activation_date', 'title', 'description']
    merge.drop(drops, axis=1, inplace=True)
    currently_unused = ['image']
    merge.drop(currently_unused, axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('data_with_fe', train, test)
else:
    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Loading')
    train, test = load_cache('data_with_fe')


print(train.shape)
print(test.shape)
print('~~~~~~~~~~~~~~~~~~~~')
print_step('Target encoding')
f_cats = ['image_top_1', 'item_seq_number', 'category_name', 'param_1', 'param_2', 'param_3']
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats, calc_std=True)
train, test = target_encode.encode(train, test, target)
print(train.shape)
print(test.shape)

print('~~~~~~~~~~~~')
print_step('Merging')
merge = pd.concat([train, test])
print(merge.shape)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('More feature encoding 1/11')
merge['price'] = np.log1p(merge['price'])
print_step('More feature encoding 2/11')
merge['image_top_1'] = train['image_top_1'].fillna(-8).astype(str)
merge['image_top_1_mean_price'] = merge.groupby('image_top_1')['price'].transform('mean')
print_step('More feature encoding 3/11')
merge['image_top_1_price_std'] = merge.groupby('image_top_1')['price'].transform('std').fillna(0)
print_step('More feature encoding 4/11')
merge['param_1_mean_price'] = merge.groupby('param_1')['price'].transform('mean')
print_step('More feature encoding 5/11')
merge['param_1_price_std'] = merge.groupby('param_1')['price'].transform('std').fillna(0)
print_step('More feature encoding 6/11')
merge['param_2_mean_price'] = merge.groupby('param_2')['price'].transform('mean')
print_step('More feature encoding 7/11')
merge['param_2_price_std'] = merge.groupby('param_2')['price'].transform('std').fillna(0)
print_step('More feature encoding 8/11')
merge['param_3_mean_price'] = merge.groupby('param_3')['price'].transform('mean')
print_step('More feature encoding 9/11')
merge['param_3_price_std'] = merge.groupby('param_3')['price'].transform('std').fillna(0)
print_step('More feature encoding 10/11')
merge['user_id_mean_price'] = merge.groupby('user_id')['price'].transform('mean')
print_step('More feature encoding 11/11')
merge['user_id_price_std'] = merge.groupby('user_id')['price'].transform('std').fillna(0)
print_step('More feature encoding 10/11')
merge['category_name_mean_price'] = merge.groupby('category_name')['price'].transform('mean')
print_step('More feature encoding 11/11')
merge['category_name_price_std'] = merge.groupby('category_name')['price'].transform('std').fillna(0)
print(merge.shape)

# print('~~~~~~~~~~~~~~')
# print_step('TFIDF 1/4')
# merge['text'] = merge['title'] + ' ' + merge['description'].fillna('')
# tfidf = TfidfVectorizer(ngram_range=(1, 2),
#                         max_features=100000,
#                         min_df=2,
#                         max_df=0.8,
#                         binary=True,
#                         encoding='KOI8-R')
# tfidf_merge = tfidf.fit_transform(merge['text'])
# print(tfidf_merge.shape)
# print_step('TFIDF 2/4')
# dim = train.shape[0]
# tfidf_train = tfidf_merge[:dim]
# tfidf_test = tfidf_merge[dim:]
# print(tfidf_train.shape)
# print(tfidf_test.shape)
# print_step('TFIDF 3/4')
# fselect = SelectKBest(f_regression, k=48000)
# tfidf_train = fselect.fit_transform(tfidf_train, target)
# tfidf_test = fselect.transform(tfidf_test)
# print_step('TFIDF 4/4')
# merge.drop('text', axis=1, inplace=True)
# print(tfidf_train.shape)
# print(tfidf_test.shape)

print(merge.shape)
print('~~~~~~~~~~~~~')
print_step('Dropping')
drops = ['user_id'] # 'text'
merge.drop(drops, axis=1, inplace=True)
print(merge.shape)

print('~~~~~~~~~~~~~~~~')
print_step('Dummies 1/2')
dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'param_1',
              'param_2', 'param_3', 'image_top_1', 'day_of_week', 'region', 'city']
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
train = merge[:dim]
test = merge[dim:]
print(train.shape)
print(test.shape)

# print('Combine')
# train = hstack((train, tfidf_train)).tocsr()
# test = hstack((test, tfidf_test)).tocsr()
# print(train.shape)
# print(test.shape)

print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train, test, target, runLGB, rmse, 'lgb')
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

# LGB: no text, geo, date, image, param data, or item_seq_number                   - Dim 51,    5CV 0.2313, Submit 0.235, Delta -.0037
# LGB: +missing data, +OHE params (no text, geo, date, image, or item_seq_number)  - Dim 5057,  5CV 0.2269, Submit 0.230, Delta -.0031
# LGB: +basic NLP (no other text, geo, date, image, or item_seq_number)            - Dim 5078,  5CV 0.2261, Submit 0.229, Delta -.0029  <a9e424c>
# LGB: +date (no other text, geo, image, or item_seq_number)                       - Dim 5086,  5CV 0.2261, Submit ?                    <f6c28f2>
# LGB: +OHE city and region (no other text, image, or item_seq_number)             - Dim 6866,  5CV 0.2254, Submit ?                    <531df17>
# LGB: +item_seq_number (no other text or image)                                   - Dim 6867,  5CV 0.2252, Submit ?                    <624f1a4>
# LGB: +more basic NLP (no other text or image)                                    - Dim 6877,  5CV 0.2251, Submit 0.229, Delta -.0039  <f47d17d>
# LGB: +SelectKBest TFIDF description + text (no image)                            - Dim 54877, 5CV 0.2221, Submit 0.225, Delta -.0029  <5e4f5be>
# LGB: -SelectKBest TFIDF, +frequency encoding                                     - Dim 6887,  5CV 0.2243, Submit ?                    <f7787b2>
# LGB: +total missing, +adjusted item_seq_number                                   - Dim 6889,  5CV 0.2243, Submit ?                    <0107d26>
# LGB: +target encoding                                                            - Dim 6901,  5CV 0.2234, Submit 0.227, Delta -.0036  <7a849b1>
# LGB: +price encoding                                                             - Dim 6912,  5CV 0.2229, Submit ?
# LGB: -                                                                           - Dim ?, 5CV ?, Submit ?

# CURRENT
# [2018-04-27 16:39:08.709221] lgb cv scores : [0.22343898332728193, 0.22250389688187794, 0.2227847395274668, 0.22261006668808794, 0.22316359778745867]
# [2018-04-27 16:39:08.710569] lgb mean cv score : 0.22290025684243467
# [2018-04-27 16:39:08.712052] lgb std cv score : 0.0003504940720958682

# [2018-04-26 21:46:32.974844] lgb cv scores : [0.22256053295460215, 0.22160517248374503, 0.22192232562406788, 0.22177930697296735, 0.22242885702632134]
# [2018-04-26 21:46:32.976301] lgb mean cv score : 0.22205923901234076
# [2018-04-26 21:46:32.979111] lgb std cv score : 0.00037180552114082565



# TODO

# mean of item_seq_number for user_id
# max of item_seq_number for user_id
# min of item_seq_number for user_id
# max - min of item_seq_number for user_id

# Get CV score re-adding TFIDF

# Handle time features
	# https://github.com/mxbi/ftim
	# OTV validation

#tr['title_first'] = tr['title'].apply(lambda ss: ss.translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').lower().split(' ')[0])
# 0.7634060532342571

#get_element = lambda elem, item: elem[item] if len(elem) > item else ''
#tr['title_second'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').lower().split(' '), 1))
# 0.7419195431638979

# Image analysis
    # pic2vec
    # img_hash.py?
	# https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16
    # Contrast? https://dsp.stackexchange.com/questions/3309/measuring-the-contrast-of-an-image
    # https://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/
    # NNs?
# Geo stuff with the cities (get lat/long)
# Train classification model with AUC
# Implement https://www.kaggle.com/yekenot/ridge-text-cat, add as submodel
# Check feature impact and tuning in DR
# Tune models some
# Vary model
    # Train Ridge on text, include into as-is LGB
    # Take LGB, add text OHE with Ridge / SelectKBest
    # Take LGB, add text OHE with SVD + embedding
    # OHE everything into Ridge and then take just encoded categorical and numeric into LGB and boost with LGB
    # OHE everything into LGB except text, then use text and residuals and boost with Ridge
# Train more models (Ridge, FM, Ridge, NNs)
# Predict log price to impute missing, also look at difference between predicted and actual price
# Look to DonorsChoose
    # https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x/code
    # https://www.kaggle.com/fizzbuzz/the-all-in-one-model
    # https://www.kaggle.com/emotionevil/beginners-workflow-meanencoding-lgb-nn-ensemble
    # https://www.kaggle.com/safavieh/ultimate-feature-engineering-xgb-lgb-nn
    # https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
    # https://www.kaggle.com/nicapotato/abc-s-of-tf-idf-boosting-0-798
# Use train_active and test_active somehow?
