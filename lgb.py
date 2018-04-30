import string

from pprint import pprint

import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
    merge['has_price'] = (merge['price'] > 0).astype(int)
    merge['image_missing'] = merge['image'].isna().astype(int)
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')
    merge['description'].fillna('', inplace=True)

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday

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

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    drops = ['activation_date', 'description', 'title']
    merge.drop(drops, axis=1, inplace=True)
    currently_unused = ['user_id', 'image']
    merge.drop(currently_unused, axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train_fe = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test_fe = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train_fe.shape)
    print(test_fe.shape)

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Title TFIDF 1/3')
    train_fe['titlecat'] = train_fe['parent_category_name'] + ' ' + train_fe['category_name'] + ' ' + train_fe['param_1'] + ' ' + train_fe['param_2'] + ' ' + train_fe['param_3'] + ' ' + train['title']
    test_fe['titlecat'] = test_fe['parent_category_name'] + ' ' + test_fe['category_name'] + ' ' + test_fe['param_1'] + ' ' + test_fe['param_2'] + ' ' + test_fe['param_3'] + ' ' + test['title']
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
    train_fe['desc'] = train['title'] + ' ' + train['description'].fillna('')
    test_fe['desc'] = test['title'] + ' ' + test['description'].fillna('')
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
    train_fe.drop(['desc', 'titlecat'], axis=1, inplace=True)
    test_fe.drop(['desc', 'titlecat'], axis=1, inplace=True)

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
submission.to_csv('submit/submit_lgb4.csv', index=False)
print_step('Done!')

# LOG (Comp start 25 Apr, merge deadline 20 June @ 7pm EDT, end 27 June @ 7pm EDT) (19/25 submits used as of 30 Apr)
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
# LGB: +Ridge Encoding title                                                       - Dim 47,    5CV 0.2205, Submit 0.223, Delta -.0025
# LGB: +                                                                           - Dim ?, 5CV ?, Submit ?

# CURRENT
# [2018-04-30 02:30:37.975538] lgb cv scores : [0.2217595472795318, 0.2207746297687226, 0.22111268905727507, 0.2208252851488291, 0.22151688106644432]
# [2018-04-30 02:30:37.975697] lgb mean cv score : 0.22119780646416057
# [2018-04-30 02:30:37.976578] lgb std cv score : 0.0003853458094671446

# [2018-04-30 11:57:26.416590] lgb cv scores : [0.22108413204150598, 0.2200999871461043, 0.2203104547955073, 0.2200721721767574, 0.2207988289043796]
# [2018-04-30 11:57:26.417441] lgb mean cv score : 0.22047311501285088
# [2018-04-30 11:57:26.419676] lgb std cv score : 0.00040146463293921745


# [100]   training's rmse: 0.223271       valid_1's rmse: 0.225434
# [200]   training's rmse: 0.220095       valid_1's rmse: 0.223689
# [300]   training's rmse: 0.218511       valid_1's rmse: 0.223022
# [400]   training's rmse: 0.217334       valid_1's rmse: 0.222639
# [500]   training's rmse: 0.216295       valid_1's rmse: 0.222348
# [600]   training's rmse: 0.215457       valid_1's rmse: 0.222149
# [700]   training's rmse: 0.214736       valid_1's rmse: 0.222021
# [800]   training's rmse: 0.2141 valid_1's rmse: 0.221919
# [900]   training's rmse: 0.213544       valid_1's rmse: 0.221831
# [1000]  training's rmse: 0.213022       valid_1's rmse: 0.22176


# TODO
# Ridge encode TFIDF of params + cats + title (careful!)
# Mean and max length of word
# Inclusion of numerics
# Include more punctuation in punctuation
# adjusted_seq_number (careful!)
# Include user_id as cat variable (careful!)
# train['user_count'] = train.groupby('user_id')['user_id'].transform('count') (careful!)
# train['num_days'] = train.groupby('user_id')['activation_date'].transform('nunique')
# train['multicat'] = train.groupby('user_id')['category_name'].transform('nunique') > 1
# train['num_img_cats'] = train.groupby('user_id')['image_top_1'].transform('nunique')
# tr['date_int'] = tr['activation_date'].astype(int); tr.groupby('user_id')['date_int'].transform('max') - tr.groupby('user_id')['date_int'].transform('min')
# tr.groupby('user_id')['date_int'].transform('std')
# tr.groupby('user_id')['date_int'].transform(lambda x: np.max(np.diff(x)) if len(x) > 1 else 0))
# Recategorize categories according to english translation (maybe by hand or CountVectorizer)
# Ridges for each parent_category

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
#tr['title_first'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').lower().split(' '), 0))
# 0.7634060532342571
#tr['title_second'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').lower().split(' '), 1))
# 0.7419195431638979
#tr['title_last'] = tr['title'].apply(lambda ss: get_element(ss.translate(ss.maketrans('', '', string.punctuation)).replace('\n', ' ').lower().split(' '), -1))
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
# Vary model
    # Train Ridge on text, include into as-is LGB
    # Take LGB, add text with Ridge / SelectKBest
    # Take LGB, add text with SVD + embedding
    # Take LGB, add text OHE with Ridge / SelectKBest
    # Take LGB, add text OHE with SVD + embedding
    # OHE everything into Ridge and then take just encoded categorical and numeric into LGB and boost with LGB
    # OHE everything into LGB except text, then use text and residuals and boost with Ridge
# Train more models (Ridge, FM, Ridge, NNs)

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
