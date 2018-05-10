import string

from pprint import pprint

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import Ridge
import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, normalize_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.05,
              'application': 'regression',
              'num_leaves': 118,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.2,
              'nthread': 3,
              'lambda_l1': 5,
              'lambda_l2': 5,
              'min_data_in_leaf': 40}
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

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Imputation 1/7')
    merge['param_1'].fillna('missing', inplace=True)
    print_step('Imputation 2/7')
    merge['param_2'].fillna('missing', inplace=True)
    print_step('Imputation 3/7')
    merge['param_3'].fillna('missing', inplace=True)
    print_step('Imputation 4/7')
    merge['price_missing'] = merge['price'].isna().astype(int)
    merge['price'].fillna(merge['price'].median(), inplace=True)
    print_step('Imputation 5/7')
    merge['image_missing'] = merge['image'].isna().astype(int)
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')
    print_step('Imputation 6/7')
    merge['description'].fillna('', inplace=True)
    print_step('Imputation 7/7')
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    merge['city'] = merge['city'] + ', ' + merge['region']

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Basic NLP 1/38')
    merge['num_words_description'] = merge['description'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 2/38')
    merge['num_words_title'] = merge['title'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 3/38')
    merge['num_chars_description'] = merge['description'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 4/38')
    merge['num_chars_title'] = merge['title'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 5/38')
    merge['num_capital_description'] = merge['description'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 6/38')
    merge['num_capital_title'] = merge['title'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 7/38')
    merge['num_lowercase_description'] = merge['description'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 8/38')
    merge['num_lowercase_title'] = merge['title'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 9/38')
    merge['capital_per_char_description'] = merge['num_capital_description'] / merge['num_chars_description']
    merge['capital_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 10/38')
    merge['capital_per_char_title'] = merge['num_capital_title'] / merge['num_chars_title']
    merge['capital_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 11/38')
    russian_punct = string.punctuation + '—»«„'
    merge['num_punctuations_description'] = merge['description'].apply(lambda x: len([c for c in str(x) if c in russian_punct])) # russian_punct has +0.00001 univariate lift over string.punctuation
    print_step('Basic NLP 12/38')
    merge['punctuation_per_char_description'] = merge['num_punctuations_description'] / merge['num_chars_description']
    merge['punctuation_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 13/38')
    merge['num_punctuations_title'] = merge['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation])) # string.punctuation has +0.0003 univariate lift over russian_punct
    print_step('Basic NLP 14/38')
    merge['punctuation_per_char_title'] = merge['num_punctuations_title'] / merge['num_chars_title']
    merge['punctuation_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 15/38')
    merge['num_words_upper_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print_step('Basic NLP 16/38')
    merge['num_words_lower_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    print_step('Basic NLP 17/38')
    merge['num_words_entitled_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print_step('Basic NLP 18/38')
    merge['chars_per_word_description'] = merge['num_chars_description'] / merge['num_words_description']
    merge['chars_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 19/38')
    merge['chars_per_word_title'] = merge['num_chars_title'] / merge['num_words_title']
    merge['chars_per_word_title'].fillna(0, inplace=True)
    print_step('Basic NLP 20/38')
    merge['description_words_per_title_words'] = merge['num_words_description'] / merge['num_words_title']
    print_step('Basic NLP 21/38')
    merge['description_words_per_title_words'].fillna(0, inplace=True)
    print_step('Basic NLP 22/38')
    merge['description_chars_per_title_chars'] = merge['num_chars_description'] / merge['num_chars_title']
    print_step('Basic NLP 23/38')
    merge['description_chars_per_title_chars'].fillna(0, inplace=True)
    print_step('Basic NLP 24/38')
    merge['num_english_chars_description'] = merge['description'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 25/38')
    merge['num_english_chars_title'] = merge['title'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 26/38')
    merge['english_chars_per_char_description'] = merge['num_english_chars_description'] / merge['num_chars_description']
    merge['english_chars_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 27/38')
    merge['english_chars_per_char_title'] = merge['num_english_chars_title'] / merge['num_chars_title']
    merge['english_chars_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 28/38')
    merge['num_english_words_description'] = merge['description'].apply(lambda ss: len([w for w in ss.lower().translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').split(' ') if all([s in string.ascii_lowercase for s in w]) and len(w) > 0]))
    print_step('Basic NLP 29/38')
    merge['english_words_per_word_description'] = merge['num_english_words_description'] / merge['num_words_description']
    merge['english_words_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 30/38')
    merge['max_word_length_description'] = merge['description'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 31/38')
    merge['max_word_length_title'] = merge['title'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 32/38')
    merge['mean_word_length_description'] = merge['description'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 33/38')
    merge['mean_word_length_title'] = merge['title'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 34/38')
    stop_words = {x: 1 for x in stopwords.words('russian')}
    merge['num_stopwords_description'] = merge['description'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    print_step('Basic NLP 35/38')
    merge['number_count_description'] = merge['description'].str.count('[0-9]')
    print_step('Basic NLP 36/38')
    merge['number_count_title'] = merge['title'].str.count('[0-9]')
    print_step('Basic NLP 37/38')
    merge['num_unique_words_description'] = merge['description'].apply(lambda x: len(set(str(x).lower().split())))
    print_step('Basic NLP 38/38')
    merge['unique_words_per_word_description'] = merge['num_unique_words_description'] / merge['num_words_description']
    merge['unique_words_per_word_description'].fillna(0, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train_fe = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test_fe = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train_fe.shape)
    print(test_fe.shape)

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('User stats 1/3 1/2')
    train_fe['adjusted_seq_num'] = train_fe['item_seq_number'] - train_fe.groupby('user_id')['item_seq_number'].transform('min')
    print_step('User stats 1/3 2/2')
    test_fe['adjusted_seq_num'] = test_fe['item_seq_number'] - test_fe.groupby('user_id')['item_seq_number'].transform('min')
    print_step('User stats 2/3 1/2')
    train_fe['user_num_days'] = train_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    print_step('User stats 2/3 2/2')
    test_fe['user_num_days'] = test_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    print_step('User stats 3/3 1/5')
    train_fe['date_int'] = pd.to_datetime(train_fe['activation_date']).astype(int)
    test_fe['date_int'] = pd.to_datetime(test_fe['activation_date']).astype(int)
    print_step('User stats 3/3 2/5')
    train_fe['user_days_range'] = train_fe.groupby('user_id')['date_int'].transform('max').astype(int) - train_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    print_step('User stats 3/3 3/5')
    test_fe['user_days_range'] = test_fe.groupby('user_id')['date_int'].transform('max').astype(int) - test_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    print_step('User stats 3/3 4/5')
    train_fe['user_days_range'] = train_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))
    print_step('User stats 3/3 5/5')
    test_fe['user_days_range'] = test_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))


    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Title TFIDF 1/2')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['title'])
    print(tfidf_train.shape)
    print_step('Title TFIDF 2/2')
    tfidf_test = tfidf.transform(test_fe['title'])
    print(tfidf_test.shape)

    print_step('Title SVD 1/4')
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Title SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Title SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Title SVD 4/4')
    train_svd.columns = ['svd_title_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_title_'+str(i+1) for i in range(10)]
    train_fe = pd.concat([train_fe, train_svd], axis=1)
    test_fe = pd.concat([test_fe, test_svd], axis=1)

    print_step('Titlecat TFIDF 1/3')
    train_fe['titlecat'] = train_fe['parent_category_name'] + ' ' + train_fe['category_name'] + ' ' + train_fe['param_1'] + ' ' + train_fe['param_2'] + ' ' + train_fe['param_3'] + ' ' + train_fe['title']
    test_fe['titlecat'] = test_fe['parent_category_name'] + ' ' + test_fe['category_name'] + ' ' + test_fe['param_1'] + ' ' + test_fe['param_2'] + ' ' + test_fe['param_3'] + ' ' + test_fe['title']
    print_step('Titlecat TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['titlecat'])
    print(tfidf_train.shape)
    print_step('Titlecat TFIDF 3/3')
    tfidf_test = tfidf.transform(test_fe['titlecat'])
    print(tfidf_test.shape)

    print_step('Titlecat SVD 1/4')
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Titlecat SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Titlecat SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Titlecat SVD 4/4')
    train_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(10)]
    train_fe = pd.concat([train_fe, train_svd], axis=1)
    test_fe = pd.concat([test_fe, test_svd], axis=1)

    print_step('Titlecat TFIDF Ridge 1/6')
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(tfidf_train, target, test_size = 0.5, shuffle = False)
    model = Ridge()
    print_step('Titlecat TFIDF Ridge 2/6 1/3')
    model.fit(X_train_1, y_train_1)
    print_step('Titlecat TFIDF Ridge 2/6 2/3')
    ridge_preds1 = model.predict(X_train_2)
    print_step('Titlecat TFIDF Ridge 2/6 3/3')
    ridge_preds1f = model.predict(tfidf_test)
    model = Ridge()
    print_step('Titlecat TFIDF Ridge 3/6 1/3')
    model.fit(X_train_2, y_train_2)
    print_step('Titlecat TFIDF Ridge 3/6 2/3')
    ridge_preds2 = model.predict(X_train_1)
    print_step('Titlecat TFIDF Ridge 3/6 3/3')
    ridge_preds2f = model.predict(tfidf_test)
    print_step('Titlecat TFIDF Ridge 4/6')
    ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
    print_step('Titlecat TFIDF Ridge 5/6')
    ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
    print_step('Titlecat Ridge RMSE OOF: {}'.format(rmse(ridge_preds_oof, target)))
    print_step('Titlecat TFIDF Ridge 6/6')
    train_fe['title_ridge'] = ridge_preds_oof
    test_fe['title_ridge'] = ridge_preds_test

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    if not is_in_cache('normalized_desc'):
        print_step('Normalize text 1/3')
        train_fe['desc'] = train_fe['title'] + ' ' + train_fe['description'].fillna('')
        test_fe['desc'] = test_fe['title'] + ' ' + test_fe['description'].fillna('')
        print_step('Normalize text 2/3')
        # HT @IggiSv9t https://www.kaggle.com/iggisv9t/handling-russian-language-inflectional-structure
        train_fe['desc'] = train_fe['desc'].astype(str).apply(normalize_text)
        print_step('Normalize text 3/3')
        test_fe['desc'] = test_fe['desc'].astype(str).apply(normalize_text)
    else:
        print_step('Loading normalized data from cache')
        normalized_desc_train, normalized_desc_test = load_cache('normalized_desc')
        train_fe['desc'] = normalized_desc_train['desc'].fillna('')
        test_fe['desc'] = normalized_desc_test['desc'].fillna('')

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Description TFIDF 1/2')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['description'])
    print(tfidf_train.shape)
    print_step('Description TFIDF 2/2')
    tfidf_test = tfidf.transform(test_fe['description'])
    print(tfidf_test.shape)

    print_step('Description SVD 1/4')
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Description SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Description SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Description SVD 4/4')
    train_svd.columns = ['svd_description_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_description_'+str(i+1) for i in range(10)]
    train_fe = pd.concat([train_fe, train_svd], axis=1)
    test_fe = pd.concat([test_fe, test_svd], axis=1)

    print_step('Text TFIDF 1/2')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['desc'])
    print(tfidf_train.shape)
    print_step('Text TFIDF 2/2')
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
    print_step('Text Ridge RMSE OOF: {}'.format(rmse(ridge_preds_oof, target)))
    print_step('Text TFIDF Ridge 6/6')
    train_fe['desc_ridge'] = ridge_preds_oof
    test_fe['desc_ridge'] = ridge_preds_test

    print('~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Title/Text TFIDF 1/3')
    train_fe['title_and_desc'] = train_fe['titlecat'] + ' ' + train_fe['description'].fillna('')
    test_fe['title_and_desc'] = test_fe['titlecat'] + ' ' + test_fe['description'].fillna('')
    print_step('Title/Text TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train_fe['title_and_desc'])
    print(tfidf_train.shape)
    print_step('Title/Text TFIDF 3/3')
    tfidf_test = tfidf.transform(test_fe['title_and_desc'])
    print(tfidf_test.shape)

    print_step('Title/Text SVD 1/4')
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Title/Text SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Title/Text SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Title/Text SVD 4/4')
    train_svd.columns = ['svd_title_and_desc_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_title_and_desc_'+str(i+1) for i in range(10)]
    train_fe = pd.concat([train_fe, train_svd], axis=1)
    test_fe = pd.concat([test_fe, test_svd], axis=1)

    print_step('Title/Text TFIDF Ridge 1/6')
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(tfidf_train, target, test_size = 0.5, shuffle = False)
    model = Ridge()
    print_step('Title/Text TFIDF Ridge 2/6 1/3')
    model.fit(X_train_1, y_train_1)
    print_step('Title/Text TFIDF Ridge 2/6 2/3')
    ridge_preds1 = model.predict(X_train_2)
    print_step('Title/Text TFIDF Ridge 2/6 3/3')
    ridge_preds1f = model.predict(tfidf_test)
    model = Ridge()
    print_step('Title/Text TFIDF Ridge 3/6 1/3')
    model.fit(X_train_2, y_train_2)
    print_step('Title/Text TFIDF Ridge 3/6 2/3')
    ridge_preds2 = model.predict(X_train_1)
    print_step('Title/Text TFIDF Ridge 3/6 3/3')
    ridge_preds2f = model.predict(tfidf_test)
    print_step('Title/Text TFIDF Ridge 4/6')
    ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
    print_step('Title/Text TFIDF Ridge 5/6')
    ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
    print_step('Title/Text Ridge RMSE OOF: {}'.format(rmse(ridge_preds_oof, target)))
    print_step('Title/Text TFIDF Ridge 6/6')
    train_fe['title_and_desc_ridge'] = ridge_preds_oof
    test_fe['title_and_desc_ridge'] = ridge_preds_test

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    drops = ['activation_date', 'description', 'title', 'desc', 'titlecat', 'title_and_desc', 'image', 'user_id', 'date_int']
    train_fe.drop(drops, axis=1, inplace=True)
    test_fe.drop(drops, axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('data_with_fe', train_fe, test_fe)
else:
    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Loading')
    train_fe, test_fe = load_cache('data_with_fe')


print('~~~~~~~~~~~~~~~~~~~~~')
print_step('Loading submodel')
train_deep_text_lgb, test_deep_text_lgb = load_cache('deep_text_lgb')
train_fe['deep_text_lgb'] = train_deep_text_lgb['deep_text_lgb']
test_fe['deep_text_lgb'] = test_deep_text_lgb['deep_text_lgb']

print('~~~~~~~~~~')
print_step('Stuff')
train_fe['price'] = train['price']
test_fe['price'] = test['price']
train_fe['price'].fillna(0, inplace=True)
test_fe['price'].fillna(0, inplace=True)
train_enc = train_fe.groupby('category_name')['price'].agg(['mean']).reset_index()
train_enc.columns = ['category_name', 'cat_price_mean']
train_fe = pd.merge(train_fe, train_enc, how='left', on='category_name')
test_fe = pd.merge(test_fe, train_enc, how='left', on='category_name')
train_fe['cat_price_diff'] = train_fe['price'] - train_fe['cat_price_mean']
test_fe['cat_price_diff'] = test_fe['price'] - test_fe['cat_price_mean']

train_fe['city'] = train_fe['city'].apply(lambda x: x.replace('_', ', '))
test_fe['city'] = test_fe['city'].apply(lambda x: x.replace('_', ', '))
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

train_enc = train_fe.groupby('parent_category_name')['parent_category_name'].agg(['count']).reset_index()
train_enc.columns = ['parent_category_name', 'parent_cat_count']
train_fe = pd.merge(train_fe, train_enc, how='left', on='parent_category_name')
test_fe = pd.merge(test_fe, train_enc, how='left', on='parent_category_name')

train_enc = train_fe.groupby('city')['city'].agg(['count']).reset_index()
train_enc.columns = ['city', 'city_count']
train_fe = pd.merge(train_fe, train_enc, how='left', on='city')
test_fe = pd.merge(test_fe, train_enc, how='left', on='city')
test_fe['city_count'].fillna(0, inplace=True)

train_fe['region_X_cat'] = train_fe['region'] + ':' + train_fe['parent_category_name']
test_fe['region_X_cat'] = test_fe['region'] + ':' + test_fe['parent_category_name']
train_enc = train_fe.groupby('region_X_cat')['region_X_cat'].agg(['count']).reset_index()
train_enc.columns = ['region_X_cat', 'region_X_cat_count']
train_fe = pd.merge(train_fe, train_enc, how='left', on='region_X_cat')
test_fe = pd.merge(test_fe, train_enc, how='left', on='region_X_cat')
train_fe.drop('region_X_cat', axis=1, inplace=True)
test_fe.drop('region_X_cat', axis=1, inplace=True)

train_fe['user_id'] = train['user_id']
test_fe['user_id'] = test['user_id']
merge = pd.concat([train_fe, test_fe])
merge['user_max_items'] = merge.groupby('user_id')['item_seq_number'].transform('max')
merge['mean_items_by_user_type'] = merge.groupby('user_type')['user_max_items'].transform('mean')
merge['user_max_items_diff'] = merge['user_max_items'] - merge['mean_items_by_user_type']
merge.drop(['mean_items_by_user_type', 'user_id'], axis=1, inplace=True)
dim = train.shape[0]
train_fe = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
test_fe = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)

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

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lgb_preds', pd.DataFrame({'lgb': results['train']}),
                           pd.DataFrame({'lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb7.csv', index=False)
print_step('Done!')

# LOG (Comp start 25 Apr, merge deadline 20 June @ 7pm EDT, end 27 June @ 7pm EDT) (26/60 submits used as of 7 May UTC) -- Average Delta 0.0035, Safety Margin 0.002
# LGB: no text, geo, date, image, param data, or item_seq_number                   - Dim 51,    5CV 0.23129, Submit 0.2355, Delta -.00421
# LGB: +missing data, +OHE params (no text, geo, date, image, or item_seq_number)  - Dim 5057,  5CV 0.22694, Submit 0.2305, Delta -.00356
# LGB: +basic NLP (no other text, geo, date, image, or item_seq_number)            - Dim 5078,  5CV 0.22607, Submit 0.2299, Delta -.00383  <a9e424c>
# LGB: +date (no other text, geo, image, or item_seq_number)                       - Dim 5086,  5CV 0.22610, Submit ?                      <f6c28f2>
# LGB: +OHE city and region (no other text, image, or item_seq_number)             - Dim 6866,  5CV 0.22540, Submit ?                      <531df17>
# LGB: +item_seq_numbur (no other text or image)                                   - Dim 6867,  5CV 0.22517, Submit ?                      <624f1a4>
# LGB: +more basic NLP (no other text or image)                                    - Dim 6877,  5CV 0.22508, Submit 0.2290, Delta -.00392  <f47d17d>
# LGB: +SelectKBest TFIDF description + text (no image)                            - Dim 54877, 5CV 0.22206, Submit 0.2257, Delta -.00364  <7002d68>
# LGB: +LGB Encode Cats and Ridge Encode text, -some missing vars, -weekend        - Dim 46,    5CV 0.22120, Submit ?
# LGB: +Ridge Encoding title                                                       - Dim 47,    5CV 0.22047, Submit 0.2237, Delta -.00323  <6a183fb>
# LGB: -some NLP +some NLP                                                         - Dim 50,    5CV 0.22040, Submit 0.2237, Delta -.00330  <954e3ad>
# LGB: +adjusted_seq_num, +user_num_days, +user_days_range                         - Dim 53,    5CV 0.22005, Submit 0.2235, Delta -.00345  <e7ea303>
# LGB: +recode city                                                                - Dim 53,    5CV 0.22006, Submit ?                      <2054ce2>
# LGB: +normalize desc                                                             - Dim 53,    5CV 0.22002, Submit ?                      <87b52f7>
# LGB: +text/title ridge                                                           - Dim 54,    5CV 0.21991, Submit ?                      <abd76a4>
# LGB: +SVD(title, 10) +SVD(description, 10) +SVD(titlecat, 10) +SVD(text/title)   - Dim 94,    5CV 0.21967, Submit 0.2230, Delta -.00333  <6e94776>
# LGB: +Deep text LGB                                                              - Dim 95,    5CV 0.21862, Submit 0.2217, Delta -.00308  <be831c5>
# LGB: +Some tuning                                                                - Dim 95,    5CV 0.21778, Submit ?0.2213?               <627d398>
# LGB: +Num unique words +Unique words ratio                                       - Dim 97,    5CV 0.21782, Submit ?0.2214?               <2bcd64e>
# LGB: +cat_price_mean +cat_price_diff                                             - Dim 99,    5CV 0.21768, Submit 0.2209, Delta -.00322  <0c9e1e4>
# LGB: +lat/lon of cities                                                          - Dim 101,   5CV 0.21766, Submit ?0.2212?               <a3d9005>
# LGB: +parent_cat_count, region_X_cat_count                                       - Dim 103,   5CV 0.21763, Submit ?0.2211?
# LGB: +city_count                                                                 - Dim 104,   5CV 0.21747, Submit ?0.2210?

# CURRENT
# [2018-05-09 22:39:42.778227] lgb cv scores : [0.21805966540045607, 0.21705929484738665, 0.21742744041563333, 0.21706854552504978, 0.2177181188036843]
# [2018-05-09 22:39:42.779709] lgb mean cv score : 0.21746661299844203
# [2018-05-09 22:39:42.781332] lgb std cv score : 0.0003849328778894197


# Title Ridge      OOF 0.2337
# Text Ridge       OOF 0.2360
# Title-Text Ridge OOF 0.2340
# Deep Text LGB    OOF 0.22196

# [100]   training's rmse: 0.217158       valid_1's rmse: 0.220805
# [200]   training's rmse: 0.213181       valid_1's rmse: 0.219241
# [300]   training's rmse: 0.2108         valid_1's rmse: 0.218764
# [400]   training's rmse: 0.208745       valid_1's rmse: 0.218526
# [500]   training's rmse: 0.207019       valid_1's rmse: 0.218346
# [600]   training's rmse: 0.205422       valid_1's rmse: 0.218263
# [700]   training's rmse: 0.204084       valid_1's rmse: 0.218185
# [800]   training's rmse: 0.202819       valid_1's rmse: 0.218122
# [900]   training's rmse: 0.201698       valid_1's rmse: 0.218091
# [1000]  training's rmse: 0.200674       valid_1's rmse: 0.21806

# TODO
# Handle price outliers
    # Call on the number that is indicated on the "price"
    # train[train.price == train.price.max()]['description'].values

# Predict log price to impute missing, also look at difference between predicted and actual price (careful!)

# Macroeconomic data for locations?

# Population encode region/city?
	# Is this different from count encoding?

# Image analysis
    # https://www.kaggle.com/peterhurford/image-feature-engineering
    # Try to get color, contrast, exposure, saturation, temperature, tint, etc. as features
        # https://dsp.stackexchange.com/questions/3309/measuring-the-contrast-of-an-image
	# https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16
	# https://www.kaggle.com/bguberfain/vgg16-train-features/code
	# https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/
    # Add pic2vec SVD to main model, make separate embedding model
    # DR pic2vec
    # Deep image model
        # https://www.kaggle.com/bguberfain/naive-lgb-with-text-images

# Char n-grams https://www.kaggle.com/c/avito-demand-prediction/discussion/56061#325063

# Start doing text embedding corrections; add SVD of embedding to main model, full embedding to OHE model, and make separate embedding model
    # https://www.kaggle.com/gunnvant/russian-word-embeddings-for-fun-and-for-profit
    # https://github.com/nlpub/russe-evaluation/tree/master/russe/measures/word2vec
    # https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5
    # https://docs.google.com/document/d/1ply0qHqUN6fumuNeJ_xAaz9kAlHyjbIU0mNDhdrFmv8/edit
# Model on combination of SVD(text), embedding, SVD(embedding) model encoding, and raw text (SelectKBest)

# Look at supplementary data

# https://github.com/mxbi/ftim

# if any lazy people used the same or similar entries for the title/description fields

# Check feature impact and tuning in DR
# Tune models some
    # Dart?

# Understand and apply https://www.kaggle.com/rdizzl3/stage-2-lgbm-stacker-8th-place-solution/code
    #tr['title_first'] = tr['title'].apply(lambda ss: ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' ')[0])
    #get_last_two = lambda elem: (elem[-2] if len(elem) >= 2 else '') + ' ' + elem[-1]
    #tr['title_last_two'] = tr['title'].apply(lambda ss: get_last_two(ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' ')))
    #get_first_two = lambda elem: elem[0] + ' ' + (elem[-2] if len(elem) >= 2 else ''))
    #tr['title_first_two'] = tr['title'].apply(lambda ss: get_first_two(ss.translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').lower().split(' ')))
    # Words in other words (e.g., param_1 or title in descripton)?

# Overall Ridge
# Ridges for each parent_category
# FM
# Check averaging vs. including for https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm/code
# Check averaging vs. including for https://www.kaggle.com/kailex/xgb-text2vec-tfidf-0-2248/code
# Check including submodels vs. including boosted model vs. averaging boosted model for https://www.kaggle.com/peterhurford/boosting-mlp-lb-0-2297/
# LibFFM
# KNN

# Classification models
    # MNB (didn't work in this model but may work in a different ensemble that can be averaged together -- worked well in Mercari)

# Can we do a two stage classification + regression?
    # Train best model as classification, try including in Regression, or using output to decide whether to send to regression versus set as 0

# Try different ensembling strategies and check feature impact using submodels and tuning in DR

# Vary model
    # Take LGB, add text with SVD + embedding
    # OHE everything into LGB except text, then use text and residuals and boost with Ridge
    # Train classification model with AUC
	# Include region_X_cat, region_X_cat_price_mean, region_X_cat_price_diff (doesn't work in current model, but has high feature importance and might work with different tuning)
	# Treat img_top_1 as numeric

# Russian NLP http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/russian-nlp

# Make NNs

# Look to DonorsChoose
    # https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x/code
    # https://www.kaggle.com/fizzbuzz/the-all-in-one-model
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
    # https://www.kaggle.com/shadowwarrior/1st-place-solution

# https://www.kaggle.com/c/avito-duplicate-ads-detection/discussion
# https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion
# https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion

# Denoising autoencoder? https://github.com/phdowling/mSDA
