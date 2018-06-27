import gc
from pprint import pprint

import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis

from sentimental import Sentimental

import pathos.multiprocessing as mp

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


params = {'learning_rate': 0.01,
          'application': 'regression',
          'num_leaves': 31,
          'verbosity': -1,
          'metric': 'rmse',
          'data_random_seed': 5,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.4,
          'nthread': mp.cpu_count(),
          'lambda_l1': 10,
          'lambda_l2': 10,
          'min_data_in_leaf': 40,
          'num_rounds': 4000,
          'verbose_eval': 100}

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
    print_step('Feature importance')
    pprint(sorted(list(zip(model.feature_importance(), train_X.columns)), reverse=True))
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/19')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/19')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 2/15')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 4/15 3/4')
train_fe = pd.concat([train_fe, train_ridge], axis=1)
print_step('Importing Data 4/15 4/4')
test_fe = pd.concat([test_fe, test_ridge], axis=1)

print_step('Importing Data 3/15 1/3')
train_base_lgb, test_base_lgb = load_cache('base_lgb')
print_step('Importing Data 3/15 2/3')
train_fe['base_lgb'] = train_base_lgb['base_lgb']
print_step('Importing Data 3/15 3/3')
test_fe['base_lgb'] = test_base_lgb['base_lgb']

print_step('Importing Data 3/15 1/3')
train_base_lgb, test_base_lgb = load_cache('base_lgb2')
print_step('Importing Data 3/15 2/3')
train_fe['base_lgb2'] = train_base_lgb['base_lgb2']
print_step('Importing Data 3/15 3/3')
test_fe['base_lgb2'] = test_base_lgb['base_lgb2']

print_step('Importing Data 3/15 1/3')
train_te_lgb, test_te_lgb = load_cache('te_lgb')
print_step('Importing Data 3/15 2/3')
train_fe['te_lgb'] = train_te_lgb['te_lgb']
print_step('Importing Data 3/15 3/3')
test_fe['te_lgb'] = test_te_lgb['te_lgb']

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb2')
print_step('Importing Data 3/15 2/3')
train_fe['te_lgb2'] = train_te_lgb2['te_lgb']
print_step('Importing Data 3/15 3/3')
test_fe['te_lgb2'] = test_te_lgb2['te_lgb']

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb3')
print_step('Importing Data 3/15 2/3')
train_fe['te_lgb3'] = train_te_lgb2['te_lgb3']
print_step('Importing Data 3/15 3/3')
test_fe['te_lgb3'] = test_te_lgb2['te_lgb3']

print_step('Importing Data 3/15 1/3')
train_base_lgb2, test_base_lgb2 = load_cache('base_lgb_poisson')
print_step('Importing Data 3/15 2/3')
train_fe['base_lgb_poisson'] = train_base_lgb2['base_lgb_poisson']
print_step('Importing Data 3/15 3/3')
test_fe['base_lgb_poisson'] = test_base_lgb2['te_lgb_poisson'] # typo

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb_poisson')
print_step('Importing Data 3/15 2/3')
train_fe['te_lgb_poisson'] = train_te_lgb2['te_lgb_poisson']
print_step('Importing Data 3/15 3/3')
test_fe['te_lgb_poisson'] = test_te_lgb2['te_lgb_poisson']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v29')
print_step('Importing Data 3/15 2/3')
train_fe['ryan_lgbm_v29'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_fe['ryan_lgbm_v29'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v33')
print_step('Importing Data 3/15 2/3')
train_fe['ryan_lgbm_v33'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_fe['ryan_lgbm_v33'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v36')
print_step('Importing Data 3/15 2/3')
train_fe['ryan_lgbm_v36'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_fe['ryan_lgbm_v36'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 4/15 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 4/15 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 4/15 3/4')
train_fe = pd.concat([train_fe, train_pcat_ridge], axis=1)
print_step('Importing Data 4/15 4/4')
test_fe = pd.concat([test_fe, test_pcat_ridge], axis=1)

print_step('Importing Data 5/15 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 5/15 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 5/15 3/4')
train_fe = pd.concat([train_fe, train_rcat_ridge], axis=1)
print_step('Importing Data 5/15 4/4')
test_fe = pd.concat([test_fe, test_rcat_ridge], axis=1)

print_step('Importing Data 6/15 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 6/15 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 6/15 3/4')
train_fe = pd.concat([train_fe, train_catb_ridge], axis=1)
print_step('Importing Data 6/15 4/4')
test_fe = pd.concat([test_fe, test_catb_ridge], axis=1)

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 7/15 2/3')
train_fe['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 7/15 3/3')
test_fe['deep_lgb'] = test_deep_lgb['deep_lgb']

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb2')
print_step('Importing Data 7/15 2/3')
train_fe['deep_lgb2'] = train_deep_lgb['deep_lgb2']
print_step('Importing Data 7/15 3/3')
test_fe['deep_lgb2'] = test_deep_lgb['deep_lgb2']

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb3')
print_step('Importing Data 7/15 2/3')
train_fe['deep_lgb3'] = train_deep_lgb['deep_lgb3']
print_step('Importing Data 7/15 3/3')
test_fe['deep_lgb3'] = test_deep_lgb['deep_lgb3']

print_step('Importing Data 8/15 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 8/15 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 8/15 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 9/15 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 9/15 2/3')
train_fe['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 9/15 3/3')
test_fe['complete_ridge'] = test_complete_ridge['complete_ridge']

print_step('Importing Data 9/15 1/3')
train_ryan_ridge, test_ryan_ridge = load_cache('ryan_ridge_sgd_v2')
print_step('Importing Data 9/15 2/3')
train_fe['ryan_ridge'] = train_ryan_ridge['oof_ridge']
print_step('Importing Data 9/15 3/3')
test_fe['ryan_ridge'] = test_ryan_ridge['oof_ridge']
print_step('Importing Data 9/15 2/3')
train_fe['ryan_sgd'] = train_ryan_ridge['oof_sgd']
print_step('Importing Data 9/15 3/3')
test_fe['ryan_sgd'] = test_ryan_ridge['oof_sgd']

print_step('Importing Data 10/15 1/3')
train_complete_fm, test_complete_fm = load_cache('complete_fm')
print_step('Importing Data 10/15 2/3')
train_fe['complete_fm'] = train_complete_fm['complete_fm']
print_step('Importing Data 10/15 3/3')
test_fe['complete_fm'] = test_complete_fm['complete_fm']

print_step('Importing Data 11/15 1/3')
train_tffm2, test_tffm2 = load_cache('tffm2')
print_step('Importing Data 11/15 2/3')
train_fe['tffm2'] = train_tffm2['tffm2']
print_step('Importing Data 11/15 3/3')
test_fe['tffm2'] = test_tffm2['tffm2']

print_step('Importing Data 12/15 1/3')
train_tffm3, test_tffm3 = load_cache('tffm3')
print_step('Importing Data 12/15 2/3')
train_fe['tffm3'] = train_tffm3['tffm3']
print_step('Importing Data 12/15 3/3')
test_fe['tffm3'] = test_tffm3['tffm3']

print_step('Importing Data 13/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_FastText')
print_step('Importing Data 13/15 2/3')
train_fe['cnn_ft'] = train_cnn_ft['CNN_FastText']
print_step('Importing Data 13/15 3/3')
test_fe['cnn_ft'] = test_cnn_ft['CNN_FastText']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_FastText_4')
print_step('Importing Data 14/15 2/3')
train_fe['cnn_ft4'] = train_cnn_ft['CNN_FastText_4']
print_step('Importing Data 14/15 3/3')
test_fe['cnn_ft4'] = test_cnn_ft['CNN_FastText_4']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('RNN_AttentionPooling')
print_step('Importing Data 14/15 2/3')
train_fe['rnn_at'] = train_cnn_ft['RNN_AttentionPooling']
print_step('Importing Data 14/15 3/3')
test_fe['rnn_at'] = test_cnn_ft['RNN_AttentionPooling']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('RNN_AttentionPooling_img2')
print_step('Importing Data 14/15 2/3')
train_fe['rnn_at2'] = train_cnn_ft['RNN_AttentionPooling_img2']
print_step('Importing Data 14/15 3/3')
test_fe['rnn_at2'] = test_cnn_ft['RNN_AttentionPooling_img2']

print_step('Importing Data 14/15 1/3')
train_cnn_ft = pd.read_csv('cache/matt_nn_oof.csv')
test_cnn_ft = pd.read_csv('cache/matt_nn_test.csv')
print_step('Importing Data 14/15 2/3')
train_fe['matt_nn'] = train_cnn_ft['matt_nn']
print_step('Importing Data 14/15 3/3')
test_fe['matt_nn'] = test_cnn_ft['deal_probability']

print_step('Importing Data 14/15 1/3')
train_multi = pd.read_csv('cache/matt_multi_nn_oof.csv')
test_multi = pd.read_csv('cache/matt_multi_nn_test.csv')
print_step('Importing Data 14/15 2/3')
train_fe = pd.concat([train_fe, train_multi], axis=1)
train_fe.drop('item_id', axis=1, inplace=True)
print_step('Importing Data 14/15 3/3')
test_fe = pd.concat([test_fe, test_multi], axis=1)
test_fe.drop('item_id', axis=1, inplace=True)

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_binary')
print_step('Importing Data 14/15 2/3')
train_fe['CNN_binary'] = train_cnn_ft['CNN_binary']
print_step('Importing Data 14/15 3/3')
test_fe['CNN_binary'] = test_cnn_ft['CNN_binary']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_binary_PL')
print_step('Importing Data 14/15 2/3')
train_fe['CNN_binary_PL'] = train_cnn_ft['CNN_binary_PL']
print_step('Importing Data 14/15 3/3')
test_fe['CNN_binary_PL'] = test_cnn_ft['CNN_binary_PL']

print_step('Importing Data 14/15 1/3')
train_liu_nn, test_liu_nn = load_cache('liu_nn')
print_step('Importing Data 14/15 2/3')
train_fe['liu_nn'] = train_liu_nn['liu_nn']
print_step('Importing Data 14/15 3/3')
test_fe['liu_nn'] = test_liu_nn['liu_nn']

print_step('Importing Data 14/15 1/3')
train_liu_nn, test_liu_nn = load_cache('liu_nn2')
print_step('Importing Data 14/15 2/3')
train_fe['liu_nn2'] = train_liu_nn['liu_nn2']
print_step('Importing Data 14/15 3/3')
test_fe['liu_nn2'] = test_liu_nn['liu_nn2']

print_step('Importing Data 14/15 1/3')
train_liu_lgb, test_liu_lgb = load_cache('liu_lgb')
print_step('Importing Data 14/15 2/3')
train_fe['liu_lgb'] = train_liu_lgb['liu_lgb']
print_step('Importing Data 14/15 3/3')
test_fe['liu_lgb'] = test_liu_lgb['liu_lgb']


models = [c for c in train_fe.columns if 'svd' not in c and 'price' not in c and 'img' not in c and 'parent_category' not in c and 'tfidf' not in c]
good_models = [m for m in models if 'lgb' in m or 'nn' in m or 'NN' in m or 'fm' in m]
pprint(sorted([(m, rmse(target, train_fe[m])) for m in good_models], key = lambda x: x[1]))
print(pd.DataFrame(np.corrcoef([train_fe[m] for m in good_models]), index = good_models, columns = good_models))

print_step('Importing Data 3/19 1/4')
train_img, test_img = load_cache('img_data')
print_step('Importing Data 3/19 2/4')
cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
train_img = train_img[cols].fillna(0)
test_img = test_img[cols].fillna(0)
print_step('Importing Data 3/19 3/4')
train_fe = pd.concat([train_fe, train_img], axis=1)
print_step('Importing Data 3/19 4/4')
test_fe = pd.concat([test_fe, test_img], axis=1)

print_step('Importing Data 6/19 1/3')
train_ecdf, test_ecdf = load_cache('price_ecdf')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_ecdf], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_ecdf], axis=1)

print_step('Importing Data 6/19 1/3')
train_price, test_price = load_cache('expected_price')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_price], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_price], axis=1)
train_fe.drop('item_id', axis=1, inplace=True)
test_fe.drop('item_id', axis=1, inplace=True)

print_step('Importing Data 6/19 1/3')
train_numeric, test_numeric = load_cache('numeric')
train_numeric.drop(['item_id', 'item_seq_number', 'price_missing'], axis=1, inplace=True)
test_numeric.drop(['item_id', 'item_seq_number', 'price_missing'], axis=1, inplace=True)
train_fe.drop(['image_top_1', 'price'], axis=1, inplace=True)
test_fe.drop(['image_top_1', 'price'], axis=1, inplace=True)
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_numeric], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_numeric], axis=1)

print_step('Importing Data 7/19 1/5')
train_active_feats, test_active_feats = load_cache('active_feats')
train_active_feats.fillna(0, inplace=True)
test_active_feats.fillna(0, inplace=True)
print_step('Importing Data 7/19 2/5')
train_fe = pd.concat([train_fe, train_active_feats], axis=1)
print_step('Importing Data 7/19 3/5')
test_fe = pd.concat([test_fe, test_active_feats], axis=1)
print_step('Importing Data 7/19 5/5')
train_fe.drop('user_id', axis=1, inplace=True)
test_fe.drop('user_id', axis=1, inplace=True)


EMBEDDING_FILE = 'cache/avito_fasttext_300d.txt'
EMBED_SIZE = 300
NCOMP = 20

def text_to_embedding(text):
    mean = np.mean([embeddings_index.get(w, np.zeros(EMBED_SIZE)) for w in text.split()], axis=0)
    if mean.shape == ():
        return np.zeros(EMBED_SIZE)
    else:
        return mean

print_step('Importing Data 11/19 1/3')
if not is_in_cache('avito_fasttext_300d'):
    print_step('Embedding 1/5')
    train, test = get_data()

    print_step('Embedding 1/5')
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    print_step('Embedding 2/5')
    train_embeddings = (train['title'].str.cat([
                            train['description'],
                        ], sep=' ', na_rep='')
                        .astype(str)
                        .fillna('missing')
                        .apply(clean_text)
                        .apply(text_to_embedding))

    print_step('Embedding 3/5')
    test_embeddings = (test['title'].str.cat([
                            test['description'],
                        ], sep=' ', na_rep='')
                        .astype(str)
                        .fillna('missing')
                        .apply(clean_text)
                        .apply(text_to_embedding))

    print_step('Embedding 4/5')
    train_embeddings_df = pd.DataFrame(train_embeddings.values.tolist(),
                                       columns = ['embed' + str(i) for i in range(EMBED_SIZE)])
    print_step('Embedding 5/5')
    test_embeddings_df = pd.DataFrame(test_embeddings.values.tolist(),
                                      columns = ['embed' + str(i) for i in range(EMBED_SIZE)])
    print_step('Caching...')
    save_in_cache('avito_fasttext_300d', train_embeddings_df, test_embeddings_df)
else:
    train_embeddings_df, test_embeddings_df = load_cache('avito_fasttext_300d')

train_fe['embedding_mean'] = train_embeddings_df.mean(axis=1)
train_fe['embedding_std'] = train_embeddings_df.std(axis=1)
train_fe['embedding_skew'] = skew(train_embeddings_df, axis=1)
train_fe['embedding_kurtosis'] = kurtosis(train_embeddings_df, axis=1)
test_fe['embedding_mean'] = test_embeddings_df.mean(axis=1)
test_fe['embedding_std'] = test_embeddings_df.std(axis=1)
test_fe['embedding_skew'] = skew(test_embeddings_df, axis=1)
test_fe['embedding_kurtosis'] = kurtosis(test_embeddings_df, axis=1)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 12/19 1/7')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1_imputed', 'day_of_week', 'img_average_color']
for col in cat_cols:
    if col in train_fe.columns:
        train_fe.drop(col, axis=1, inplace=True)
        test_fe.drop(col, axis=1, inplace=True)
print_step('Importing Data 12/19 2/7')
train_target_encoding, test_target_encoding = load_cache('target_encoding_1000')
print_step('Importing Data 12/19 3/7')
train_fe = pd.concat([train_fe, train_target_encoding], axis=1)
print_step('Importing Data 12/19 4/7')
test_fe = pd.concat([test_fe, test_target_encoding], axis=1)
print_step('Importing Data 12/19 5/7')
train_nb, test_nb = load_cache('naive_bayes_svd_10')
print_step('Importing Data 12/19 6/7')
train_fe = pd.concat([train_fe, train_nb], axis=1)
print_step('Importing Data 12/19 7/7')
test_fe = pd.concat([test_fe, test_nb], axis=1)


print_step('Embedding SVD 1/4')
svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
svd.fit(train_embeddings_df)
print_step('Embedding SVD 2/4')
train_svd = pd.DataFrame(svd.transform(train_embeddings_df))
print_step('Embedding SVD 3/4')
test_svd = pd.DataFrame(svd.transform(test_embeddings_df))
print_step('Embedding SVD 4/4')
train_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
test_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
train_fe = pd.concat([train_fe, train_svd], axis=1)
test_fe = pd.concat([test_fe, test_svd], axis=1)

print('~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Flat Blender LGB')
print(train_fe.shape)
print(test_fe.shape)
results = run_cv_model(train_fe, test_fe, target, runLGB, params, rmse, 'flat_blender_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('flat_blender_lgb', pd.DataFrame({'flat_blender_lgb': results['train']}),
                                  pd.DataFrame({'flat_blender_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_flat_blender_lgb.csv', index=False)
