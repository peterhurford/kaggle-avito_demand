import string

from pprint import pprint

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
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
              'bagging_fraction': 0.9,
              'feature_fraction': 0.4,
              'nthread': 3,
              'lambda_l1': 1,
              'lambda_l2': 1}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=1000,
                      valid_sets=watchlist,
                      verbose_eval=100)
    print_step('Feature importance')
    pprint(sorted(list(zip(model.feature_importance(), train_X.columns)), reverse=True))
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
    print_step('Imputation')
    merge['param_1'].fillna('missing', inplace=True)
    merge['param_2'].fillna('missing', inplace=True)
    merge['param_3'].fillna('missing', inplace=True)
    merge['price_missing'] = merge['price'].isna().astype(int)
    merge['price'].fillna(merge['price'].median(), inplace=True)
    merge['image_missing'] = merge['image'].isna().astype(int)
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')
    merge['description'].fillna('', inplace=True)

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Basic NLP 1/36')
    merge['num_words_description'] = merge['description'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 2/36')
    merge['num_words_title'] = merge['title'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 3/36')
    merge['num_chars_description'] = merge['description'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 4/36')
    merge['num_chars_title'] = merge['title'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 5/36')
    merge['num_capital_description'] = merge['description'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 6/36')
    merge['num_capital_title'] = merge['title'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 7/36')
    merge['num_lowercase_description'] = merge['description'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 8/36')
    merge['num_lowercase_title'] = merge['title'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 9/36')
    merge['capital_per_char_description'] = merge['num_capital_description'] / merge['num_chars_description']
    merge['capital_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 10/36')
    merge['capital_per_char_title'] = merge['num_capital_title'] / merge['num_chars_title']
    merge['capital_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 11/36')
    russian_punct = string.punctuation + '—»«„'
    merge['num_punctuations_description'] = merge['description'].apply(lambda x: len([c for c in str(x) if c in russian_punct])) # russian_punct has +0.00001 univariate lift over string.punctuation
    print_step('Basic NLP 12/36')
    merge['punctuation_per_char_description'] = merge['num_punctuations_description'] / merge['num_chars_description']
    merge['punctuation_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 13/36')
    merge['num_punctuations_title'] = merge['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation])) # string.punctuation has +0.0003 univariate lift over russian_punct
    print_step('Basic NLP 14/36')
    merge['punctuation_per_char_title'] = merge['num_punctuations_title'] / merge['num_chars_title']
    merge['punctuation_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 15/36')
    merge['num_words_upper_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print_step('Basic NLP 16/36')
    merge['num_words_lower_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    print_step('Basic NLP 17/36')
    merge['num_words_entitled_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print_step('Basic NLP 18/36')
    merge['chars_per_word_description'] = merge['num_chars_description'] / merge['num_words_description']
    merge['chars_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 19/36')
    merge['chars_per_word_title'] = merge['num_chars_title'] / merge['num_words_title']
    merge['chars_per_word_title'].fillna(0, inplace=True)
    print_step('Basic NLP 20/36')
    merge['description_words_per_title_words'] = merge['num_words_description'] / merge['num_words_title']
    print_step('Basic NLP 21/36')
    merge['description_words_per_title_words'].fillna(0, inplace=True)
    print_step('Basic NLP 22/36')
    merge['description_chars_per_title_chars'] = merge['num_chars_description'] / merge['num_chars_title']
    print_step('Basic NLP 23/36')
    merge['description_chars_per_title_chars'].fillna(0, inplace=True)
    print_step('Basic NLP 24/36')
    merge['num_english_chars_description'] = merge['description'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 25/36')
    merge['num_english_chars_title'] = merge['title'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 26/36')
    merge['english_chars_per_char_description'] = merge['num_english_chars_description'] / merge['num_chars_description']
    merge['english_chars_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 27/36')
    merge['english_chars_per_char_title'] = merge['num_english_chars_title'] / merge['num_chars_title']
    merge['english_chars_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 28/36')
    merge['num_english_words_description'] = merge['description'].apply(lambda ss: len([w for w in ss.lower().translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').split(' ') if all([s in string.ascii_lowercase for s in w]) and len(w) > 0]))
    print_step('Basic NLP 29/36')
    merge['english_words_per_char_description'] = merge['num_english_words_description'] / merge['num_words_description']
    merge['english_words_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 30/36')
    merge['max_word_length_description'] = merge['description'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 31/36')
    merge['max_word_length_title'] = merge['title'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 32/36')
    merge['mean_word_length_description'] = merge['description'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 33/36')
    merge['mean_word_length_title'] = merge['title'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 34/36')
    stop_words = {x: 1 for x in stopwords.words('russian')}
    merge['num_stopwords_description'] = merge['description'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    print_step('Basic NLP 35/36')
    merge['number_count_description'] = merge['description'].str.count('[0-9]')
    print_step('Basic NLP 36/36')
    merge['number_count_title'] = merge['title'].str.count('[0-9]')

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train_fe = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test_fe = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train_fe.shape)
    print(test_fe.shape)

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('User stats 1/2')
    train_fe['adjusted_seq_num'] = train_fe['item_seq_number'] - train_fe.groupby('user_id')['item_seq_number'].transform('min')
    test_fe['adjusted_seq_num'] = test_fe['item_seq_number'] - test_fe.groupby('user_id')['item_seq_number'].transform('min')
    print_step('User stats 2/2')
    train_fe['user_num_days'] = train_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    test_fe['user_num_days'] = test_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    print_step('User stats 3/3')
    train_fe['date_int'] = pd.to_datetime(train_fe['activation_date']).astype(int)
    test_fe['date_int'] = pd.to_datetime(test_fe['activation_date']).astype(int)
    train_fe['user_days_range'] = train_fe.groupby('user_id')['date_int'].transform('max').astype(int) - train_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    test_fe['user_days_range'] = test_fe.groupby('user_id')['date_int'].transform('max').astype(int) - test_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    train_fe['user_days_range'] = train_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))
    test_fe['user_days_range'] = test_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))


    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Title TFIDF 1/3')
    train_fe['titlecat'] = train_fe['parent_category_name'] + ' ' + train_fe['category_name'] + ' ' + train_fe['param_1'] + ' ' + train_fe['param_2'] + ' ' + train_fe['param_3'] + ' ' + train_fe['title']
    test_fe['titlecat'] = test_fe['parent_category_name'] + ' ' + test_fe['category_name'] + ' ' + test_fe['param_1'] + ' ' + test_fe['param_2'] + ' ' + test_fe['param_3'] + ' ' + test_fe['title']
    print_step('Title TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['titlecat'])
    print(tfidf_train.shape)
    print_step('Title TFIDF 3/3')
    tfidf_test = tfidf.transform(test_fe['titlecat'])
    print(tfidf_test.shape)

    print_step('Title TFIDF Ridge 1/6')
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(tfidf_train, target, test_size = 0.5, shuffle = False)
    model = Ridge()
    print_step('Title TFIDF Ridge 2/6 1/3')
    model.fit(X_train_1, y_train_1)
    print_step('Title TFIDF Ridge 2/6 2/3')
    ridge_preds1 = model.predict(X_train_2)
    print_step('Title TFIDF Ridge 2/6 3/3')
    ridge_preds1f = model.predict(tfidf_test)
    model = Ridge()
    print_step('Title TFIDF Ridge 3/6 1/3')
    model.fit(X_train_2, y_train_2)
    print_step('Title TFIDF Ridge 3/6 2/3')
    ridge_preds2 = model.predict(X_train_1)
    print_step('Title TFIDF Ridge 3/6 3/3')
    ridge_preds2f = model.predict(tfidf_test)
    print_step('Title TFIDF Ridge 4/6')
    ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
    print_step('Title TFIDF Ridge 5/6')
    ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
    print_step('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, target)))
    print_step('Title TFIDF Ridge 6/6')
    train_fe['title_ridge'] = ridge_preds_oof
    test_fe['title_ridge'] = ridge_preds_test

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Text TFIDF 1/3')
    train_fe['desc'] = train_fe['title'] + ' ' + train_fe['description'].fillna('')
    test_fe['desc'] = test_fe['title'] + ' ' + test_fe['description'].fillna('')
    print_step('Text TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['desc'])
    print(tfidf_train.shape)
    print_step('Text TFIDF 3/3')
    tfidf_test = tfidf.transform(test_fe['desc'])
    print(tfidf_test.shape)

    print_step('Text TFIDF Ridge 1/6')
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(tfidf_train, target, test_size = 0.5, shuffle = False)
    model = Ridge()
    print_step('Text TFIDF Ridge 2/6 1/3')
    model.fit(X_train_1, y_train_1)
    print_step('Text TFIDF Ridge 2/6 2/3')
    ridge_preds1 = model.predict(X_train_2)
    print_step('Text TFIDF Ridge 2/6 3/3')
    ridge_preds1f = model.predict(tfidf_test)
    model = Ridge()
    print_step('Text TFIDF Ridge 3/6 1/3')
    model.fit(X_train_2, y_train_2)
    print_step('Text TFIDF Ridge 3/6 2/3')
    ridge_preds2 = model.predict(X_train_1)
    print_step('Text TFIDF Ridge 3/6 3/3')
    ridge_preds2f = model.predict(tfidf_test)
    print_step('Text TFIDF Ridge 4/6')
    ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
    print_step('Text TFIDF Ridge 5/6')
    ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
    print_step('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, target)))
    print_step('Text TFIDF Ridge 6/6')
    train_fe['desc_ridge'] = ridge_preds_oof
    test_fe['desc_ridge'] = ridge_preds_test

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    drops = ['activation_date', 'description', 'title', 'desc', 'titlecat', 'image', 'user_id', 'date_int']
    train_fe.drop(drops, axis=1, inplace=True)
    test_fe.drop(drops, axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('data_with_fe', train_fe, test_fe)
else:
    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Loading')
    train_fe, test_fe = load_cache('data_with_fe')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
train_fe['image_top_1'] = train_fe['image_top_1'].astype('str').fillna('missing')
test_fe['image_top_1'] = test_fe['image_top_1'].astype('str').fillna('missing')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'day_of_week']
for col in train_fe.columns:
    print(col)
    if col in cat_cols:
        train_fe[col] = train_fe[col].astype('category')
        test_fe[col] = test_fe[col].astype('category')
    else:
        train_fe[col] = train_fe[col].astype(np.float64)
        test_fe[col] = test_fe[col].astype(np.float64)


print('~~~~~~~~~~~~~~~~~~~~~~')
print_step('Pre-flight checks')
print('-')
print(train_fe.shape)
print(test_fe.shape)
print('-')
print(train_fe.columns)
print(test_fe.columns)
print('-')
print(train_fe.dtypes)
print(test_fe.dtypes)
print('-')
print(train_fe.isna().sum())
print(test_fe.isna().sum())
print('-')
print(train_fe.apply(lambda c: c.nunique()))
print(test_fe.apply(lambda c: c.nunique()))
print('-')
for col in train_fe.columns:
    print('##')
    print(col)
    print('-')
    print(train_fe[col].values)
    print('-')
    print(test_fe[col].values)
    print('-')
print('-')


print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'lgb')
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
submission.to_csv('submit/submit_lgb6.csv', index=False)
print_step('Done!')

# LOG (Comp start 25 Apr, merge deadline 20 June @ 7pm EDT, end 27 June @ 7pm EDT) (23/30 submits used as of 1 May UTC)
# LGB: no text, geo, date, image, param data, or item_seq_number                   - Dim 51,    5CV 0.2313, Submit 0.235, Delta -.0037
# LGB: +missing data, +OHE params (no text, geo, date, image, or item_seq_number)  - Dim 5057,  5CV 0.2269, Submit 0.230, Delta -.0031
# LGB: +basic NLP (no other text, geo, date, image, or item_seq_number)            - Dim 5078,  5CV 0.2261, Submit 0.229, Delta -.0029  <a9e424c>
# LGB: +date (no other text, geo, image, or item_seq_number)                       - Dim 5086,  5CV 0.2261, Submit ?                    <f6c28f2>
# LGB: +OHE city and region (no other text, image, or item_seq_number)             - Dim 6866,  5CV 0.2254, Submit ?                    <531df17>
# LGB: +item_seq_number (no other text or image)                                   - Dim 6867,  5CV 0.2252, Submit ?                    <624f1a4>
# LGB: +more basic NLP (no other text or image)                                    - Dim 6877,  5CV 0.2251, Submit 0.229, Delta -.0039  <f47d17d>
# LGB: +SelectKBest TFIDF description + text (no image)                            - Dim 54877, 5CV 0.2221, Submit 0.225, Delta -.0029  <7002d68>
# LGB: +LGB Encoding Categoricals and Ridge Encoding text                          - Dim 51,    5CV 0.2212, Submit 0.224, Delta -.0028  <e1952cf>
# LGB: -some missing vars, -weekend                                                - Dim 46,    5CV 0.2212, Submit ?
# LGB: +Ridge Encoding title                                                       - Dim 47,    5CV 0.2205, Submit 0.223, Delta -.0025  <6a183fb>
# LGB: -some NLP +some NLP                                                         - Dim 50,    5CV 0.2204, Submit 0.223, Delta -.0026  <954e3ad>
# LGB: +adjusted_seq_num                                                           - Dim 51,    5CV 0.2204, Submit 0.223, Delta -.0026
# LGB: +user_num_days, +user_days_range                                            - Dim 53,    5CV 0.2201, Submit 0.223, Delta -.0029

# CURRENT
# [2018-05-01 00:27:20.887344] lgb cv scores : [0.22056615350576797, 0.21973502608922202, 0.2199345256183117, 0.2196796908680229, 0.2203798293579068]
# [2018-05-01 00:27:20.890142] lgb mean cv score : 0.22005904508784627
# [2018-05-01 00:27:20.892303] lgb std cv score : 0.00035340190415608956


# [100]   training's rmse: 0.222022       valid_1's rmse: 0.224351
# [200]   training's rmse: 0.219022       valid_1's rmse: 0.222628
# [300]   training's rmse: 0.217297       valid_1's rmse: 0.221893
# [400]   training's rmse: 0.216177       valid_1's rmse: 0.221501
# [500]   training's rmse: 0.215132       valid_1's rmse: 0.221232
# [600]   training's rmse: 0.214295       valid_1's rmse: 0.221033
# [700]   training's rmse: 0.213495       valid_1's rmse: 0.220901
# [800]   training's rmse: 0.212805       valid_1's rmse: 0.220777
# [900]   training's rmse: 0.212164       valid_1's rmse: 0.220668
# [1000]  training's rmse: 0.211577       valid_1's rmse: 0.220566


# TODO
# Recategorize categories according to english translation (maybe by hand or CountVectorizer)

# Ridges for each parent_category
# FM model
# Deep LGB model with TFIDF

# if any lazy people used the same or similar entries for the title/description fields

# Handle time features
       # https://github.com/mxbi/ftim
       # Try OTV validation
    # Include date as a feature
    # Days since user last posted and such

# Look for covariate shifts

# Population encode region/city? (careful!)
# Geo encode region/city? (careful!)

# Handle price outliers
# Look at difference between log price and log mean price by category, user, region, category X region (careful!)
# Predict log price to impute missing, also look at difference between predicted and actual price (careful!)

# Handle russian inflectional structure <https://www.kaggle.com/iggisv9t/handling-russian-language-inflectional-structure>
# Russian NLP http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/russian-nlp

# Try Title SVD + Desc SVD vs. Text SVD vs. Title SVD + Desc SVD + Text SVD
# Add Embedding and start doing embedding corrections
    # https://github.com/nlpub/russe-evaluation/tree/master/russe/measures/word2vec
# See if SVD + embedding + top 300 words

#get_element = lambda elem, item: elem[item] if len(elem) > item else ''
#tr['title_first'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' '), 0))
# 0.7634060532342571
#tr['title_second'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' '), 1))
# 0.7419195431638979
#tr['title_last'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' '), -1))
# 0.7815870814266627

# Understand and apply https://www.kaggle.com/rdizzl3/stage-2-lgbm-stacker-8th-place-solution/code

# Category - region interaction?
# Look at user_ids that are in both train and test (careful!)
# Delta between price and price of category (careful!)

# Translate to english?
# Words in other words (e.g., param_1 or title in descripton)?

# Image analysis
    # https://www.kaggle.com/wesamelshamy/image-classification-and-quality-score-w-resnet50
    # pic2vec
    # img_hash.py?
    # https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16
    # https://www.kaggle.com/bguberfain/vgg16-train-features/code
    # Contrast? https://dsp.stackexchange.com/questions/3309/measuring-the-contrast-of-an-image
    # https://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/
    # NNs?

# Train classification model with AUC

# Owe two novel contributions in kernels (pay it forward for images, Russian NLP)

# Check feature impact and tuning in DR
# Tune models some
# Train more models (Ridge, FM, MNB, Deep LGB, KNN, NNs)
# Vary model
    # Train Ridge on text, include into as-is LGB
    # Take LGB, add text with Ridge / SelectKBest
    # Take LGB, add text with SVD + embedding
    # Take LGB, add text OHE with Ridge / SelectKBest
    # Take LGB, add text OHE with SVD + embedding
    # OHE everything into Ridge and then take just encoded categorical and numeric into LGB and boost with LGB
    # OHE everything into LGB except text, then use text and residuals and boost with Ridge

# Look to DonorsChoose
    # https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x/code
    # https://www.kaggle.com/fizzbuzz/the-all-in-one-model
    # https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
    # https://www.kaggle.com/nicapotato/abc-s-of-tf-idf-boosting-0-798
    # https://www.kaggle.com/shadowwarrior/1st-place-solution

# https://www.kaggle.com/c/avito-duplicate-ads-detection/discussion
# https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion
# https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion

# Use train_active and test_active somehow?
# Denoising autoencoder? https://github.com/phdowling/mSDA
