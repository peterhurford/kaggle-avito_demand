# This is a LightGBM Kernel that demonstrates how to combine target encoding,
# frequency encoding, price encoding, and text analysis with TFIDF. It shows
# how to do a three-fold cross validation, which is a Kaggle best practice,
# especially when ensembling.

import string

from datetime import datetime
from math import sqrt

import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score

import lightgbm as lgb


def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~')
print_step('Merging')
merge = pd.concat([train, test])

print('~~~~~~~~~~~~~~~~')
print_step('Make target')
train['target'] = 0
test['target'] = 1
merge = pd.concat([train, test])
target = merge['target']
merge.drop('target', axis=1, inplace=True)

print('~~~~~~~~~~~~~~~')
print_step('Impute 1/7')
merge['adjusted_item_seq_number'] = merge['item_seq_number'] - merge.groupby('user_id')['item_seq_number'].transform('min')
merge['num_missing'] = merge.isna().sum(axis=1)
print_step('Impute 2/7')
merge['param_1_missing'] = merge['param_1'].isna().astype(int)
merge['param_1'].fillna('missing', inplace=True)
print_step('Impute 3/7')
merge['param_2_missing'] = merge['param_2'].isna().astype(int)
merge['param_2'].fillna('missing', inplace=True)
print_step('Impute 4/7')
merge['param_3_missing'] = merge['param_3'].isna().astype(int)
merge['param_3'].fillna('missing', inplace=True)
print_step('Impute 5/7')
merge['price_missing'] = merge['price'].isna().astype(int)
merge['price'].fillna(merge['price'].median(), inplace=True)
merge['has_price'] = (merge['price'] > 0).astype(int)
print_step('Impute 6/7')
merge['description_missing'] = merge['description'].isna().astype(int)
merge['description'].fillna('', inplace=True)
print_step('Impute 7/7')
merge['image_top_1'] = train['image_top_1'].fillna(-8).astype(str)

print('~~~~~~~~~~~~~~~~~~')
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

print('~~~~~~~~~~~~~')
print_step('Dropping')
drops = ['activation_date', 'title', 'description', 'user_id']
merge.drop(drops, axis=1, inplace=True)
currently_unused = ['image'] # TODO: Don't yet know how to effectively use images
merge.drop(currently_unused, axis=1, inplace=True)
print(merge.shape)

def ohe_column(merge, col):
    ohe_df = pd.get_dummies(merge[col])
    ohe_df.columns = [col + '_' + str(c) for c in ohe_df.columns]
    merge.drop(col, axis=1, inplace=True)
    merge = pd.concat([merge, ohe_df], axis=1)
    return merge

print_step('Dummies')
dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'param_1',
              'param_2', 'param_3', 'image_top_1', 'day_of_week', 'region', 'city']
for col in dummy_cols:
    print('...' + col)
    merge = ohe_column(merge, col)
print(merge.shape)


def run_cv_model(train, target, model_fn, eval_fn, label, cols):
    """
    Takes datasets and a model and runs four-fold cross validation.
    
    `train` and `test` are numpy arrays for the train and test set.
    `target` is a numpy array specifying the target.
    `model_fn` is a function that runs a predictive model. `model_fn`
        must take the train set, train target, validation set,
        validation target, and test set as parameters, in that order.
    `eval_fn` is a function to take actual values and predictions, in that
        order, and return an evaluation score.
    `label` is a human readable label that is printed out.
    """
    kf = KFold(n_splits=3, shuffle=True, random_state=2017)
    fold_splits = kf.split(train)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    i = 1
    for dev_index, val_index in fold_splits:
        print_step('Started ' + label + ' fold ' + str(i) + '/3')
        dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, cols)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_score = eval_fn(val_y, pred_val_y)
        cv_scores.append(eval_fn(val_y, pred_val_y))
        print_step(label + ' cv score ' + str(i) + ' : ' + str(cv_score))
        i += 1
    print_step(label + ' cv scores : ' + str(cv_scores))
    print_step(label + ' mean cv score : ' + str(np.mean(cv_scores)))
    print_step(label + ' std cv score : ' + str(np.std(cv_scores)))
    pred_full_test = pred_full_test / 3.0
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores}
    return results


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, cols):
    """
    Function to run LightGBM within `run_cv_model`.
    """
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.1,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.8,
              'nthread': 3}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=1000,
                      valid_sets=watchlist,
                      verbose_eval=100)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    import pdb
    pdb.set_trace()
    return pred_test_y

print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train=merge.values,
                       target=target,
                       model_fn=runLGB,
                       eval_fn=roc_auc_score,
                       cols=merge.columns,
                       label='lgb')
import pdb
pdb.set_trace()
