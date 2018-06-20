import gc
from pprint import pprint

import pandas as pd

from nltk.corpus import stopwords

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, bin_and_ohe_data
from cache import get_data, is_in_cache, load_cache, save_in_cache


params = {'learning_rate': 0.05,
          'application': 'regression',
          'max_depth': 9,
          'num_leaves': 300,
          'verbosity': -1,
          'metric': 'rmse',
          'data_random_seed': 4,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.4,
          'nthread': 16,
          'lambda_l1': 1,
          'lambda_l2': 1,
          'min_data_in_leaf': 40,
          'num_rounds': 4800,
          'verbose_eval': 10}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print_step('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/13')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

if not is_in_cache('title_countvec'):
    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Title CountVec 1/2')
    cv = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=2)
    tfidf_train = cv.fit_transform(train['title'])
    print(tfidf_train.shape)
    print_step('Title CountVec 2/2')
    tfidf_test = cv.transform(test['title'])
    print(tfidf_test.shape)
    print_step('Saving to cache...')
    save_in_cache('title_countvec', tfidf_train, tfidf_test)

if not is_in_cache('deep_text_feats2'):
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data 2/13')
    tfidf_train, tfidf_test = load_cache('title_countvec')

    print_step('Importing Data 3/13')
    tfidf_train2, tfidf_test2 = load_cache('text_tfidf')

    print_step('Importing Data 4/13')
    tfidf_train3, tfidf_test3 = load_cache('text_char_tfidf')


    print_step('Importing Data 5/13')
    train = hstack((tfidf_train2, tfidf_train3)).tocsr()
    print_step('Importing Data 6/13')
    test = hstack((tfidf_test2, tfidf_test3)).tocsr()
    print(train.shape)
    print(test.shape)

    print_step('SelectKBest 1/2')
    fselect = SelectKBest(f_regression, k=100000)
    train = fselect.fit_transform(train, target)
    print_step('SelectKBest 2/2')
    test = fselect.transform(test)
    print(train.shape)
    print(test.shape)

    print_step('Importing Data 7/13')
    train = hstack((tfidf_train, train)).tocsr()
    print_step('Importing Data 8/13')
    test = hstack((tfidf_test, test)).tocsr()
    print(train.shape)
    print(test.shape)

    print_step('GC')
    del tfidf_test
    del tfidf_test2
    del tfidf_test3
    del tfidf_train
    del tfidf_train2
    del tfidf_train3
    gc.collect()

    print_step('Importing Data 9/13')
    train_fe, test_fe = load_cache('data_with_fe')
    dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1',
                  'day_of_week', 'region', 'city', 'param_1', 'param_2', 'param_3', 'cat_bin']
    numeric_cols = ['price', 'num_words_description', 'num_words_title', 'num_chars_description',
                    'num_chars_title', 'num_capital_description', 'num_capital_title', 'num_lowercase_title',
                    'capital_per_char_description', 'capital_per_char_title', 'num_punctuations_description',
                    'punctuation_per_char_description', 'punctuation_per_char_title', 'num_words_upper_description',
                    'num_words_lower_description', 'num_words_entitled_description', 'chars_per_word_description',
                    'chars_per_word_title', 'description_words_per_title_words', 'description_chars_per_title_chars',
                    'num_english_chars_description', 'num_english_chars_title', 'english_chars_per_char_description',
                    'english_chars_per_char_title', 'num_english_words_description', 'english_words_per_word_description',
                    'max_word_length_description', 'max_word_length_title', 'mean_word_length_description', 'mean_word_length_title',
                    'num_stopwords_description', 'number_count_description', 'number_count_title', 'num_unique_words_description',
                    'unique_words_per_word_description', 'item_seq_number', 'adjusted_seq_num', 'user_num_days', 'user_days_range',
                    'cat_price_mean', 'cat_price_diff', 'parent_cat_count', 'region_X_cat_count', 'city_count',
					'num_lowercase_description', 'num_punctuations_title', 'sentence_mean', 'sentence_std',
					'words_per_sentence', 'param_2_price_mean', 'param_2_price_diff', 'image_top_1_price_mean',
                    'image_top_1_price_diff']

    print_step('Importing Data 10/13 1/3')
    train_img, test_img = load_cache('img_data')
    print_step('Importing Data 10/13 2/3')
    drops = ['item_id', 'img_path', 'img_std_color', 'img_sum_color', 'img_rms_color',
             'img_var_color', 'img_average_color', 'deal_probability']
    drops += [c for c in train_img if 'hist' in c]
    img_dummy_cols = ['img_average_color']
    img_numeric_cols = list(set(train_img.columns) - set(drops) - set(dummy_cols))
    print_step('Importing Data 10/13 3/3')
    train_img = train_img[img_numeric_cols + img_dummy_cols].fillna(0)
    test_img = test_img[img_numeric_cols + img_dummy_cols].fillna(0)
    dummy_cols += img_dummy_cols
    numeric_cols += img_numeric_cols

    print_step('Importing Data 11/13 1/3')
    # HT: https://www.kaggle.com/jpmiller/russian-cities/data
    # HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
    locations = pd.read_csv('city_latlons.csv')
    print_step('Importing Data 11/13 2/3')
    train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
    print_step('Importing Data 11/13 3/3')
    test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
    numeric_cols += ['lat', 'lon']

    print_step('Importing Data 12/13 1/3')
    region_macro = pd.read_csv('region_macro.csv')
    print_step('Importing Data 12/13 2/3')
    train_fe = train_fe.merge(region_macro, how='left', on='region')
    print_step('Importing Data 12/13 3/3')
    test_fe = test_fe.merge(region_macro, how='left', on='region')
    numeric_cols += ['unemployment_rate', 'GDP_PC_PPP', 'HDI']

    print_step('Importing Data 13/13 1/4')
    train_active, test_active = load_cache('active_feats')
    print_step('Importing Data 13/13 2/4')
    train_active.fillna(0, inplace=True)
    test_active.fillna(0, inplace=True)
    print_step('Importing Data 13/13 3/4')
    train_active.drop('user_id', axis=1, inplace=True)
    test_active.drop('user_id', axis=1, inplace=True)
    print_step('Importing Data 13/13 4/4')
    numeric_cols += train_active.columns.values.tolist()

    print_step('CSR 1/7')
    train_ = pd.concat([train_fe, train_img, train_active], axis=1)
    print_step('CSR 2/7')
    test_ = pd.concat([test_fe, test_img, test_active], axis=1)
    print_step('CSR 3/7')
    train_ohe = csr_matrix(train_[numeric_cols])
    print_step('CSR 4/7')
    test_ohe = csr_matrix(test_[numeric_cols])
    print_step('CSR 5/7')
    train_ohe2, test_ohe2 = bin_and_ohe_data(train_, test_, numeric_cols=[], dummy_cols=dummy_cols)
    print_step('CSR 6/7')
    train = hstack((train, train_ohe, train_ohe2)).tocsr()
    print(train.shape)
    print_step('CSR 7/7')
    test = hstack((test, test_ohe, test_ohe2)).tocsr()
    print(test.shape)

    print_step('GC')
    del train_fe
    del test_fe
    del train_img
    del test_img
    del train_active
    del test_active
    del train_
    del test_
    del test_ohe
    del test_ohe2
    del train_ohe
    del train_ohe2
    gc.collect()

    print_step('Caching')
    save_in_cache('deep_text_feats2', train, test)
else:
    train, test = load_cache('deep_text_feats2')


print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train, test, target, runLGB, params, rmse, 'deep_lgb2')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('deep_lgb2', pd.DataFrame({'deep_lgb2': results['train']}),
                           pd.DataFrame({'deep_lgb2': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_deep_lgb2.csv', index=False)
print_step('Done!')
