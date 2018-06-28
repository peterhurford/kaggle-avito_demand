import gc

from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache

params = {'learning_rate': 0.02,
          'application': 'regression',
          'num_leaves': 31,
          'verbosity': -1,
          'metric': 'rmse',
          'data_random_seed': 3,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.8,
          'nthread': mp.cpu_count(),
          'lambda_l1': 1,
          'lambda_l2': 1,
          'min_data_in_leaf': 40,
          'verbose_eval': 20,
          'num_rounds': 1600}
poisson_params = params.copy()
poisson_params['application'] = 'poisson'
poisson_params['poisson_max_delta_step'] = 1.5
poisson_params['num_rounds'] = 8000

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
print_step('Importing Data 1/15')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/15')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 2/15')
train_ridge, test_ridge = load_cache('tfidf_ridges')
drops = [c for c in train_ridge.columns if 'svd' in c or 'tfidf' in c]
train_ridge.drop(drops, axis=1, inplace=True)
test_ridge.drop(drops, axis=1, inplace=True)
train_ = train_ridge
test_ = test_ridge

print_step('Importing Data 2/15')
train_['parent_category_name'] = train_fe['parent_category_name']
test_['parent_category_name'] = test_fe['parent_category_name']
train_['price'] = train_fe['price']
test_['price'] = test_fe['price']

print_step('Importing Data 3/15 1/3')
train_base_lgb, test_base_lgb = load_cache('base_lgb')
print_step('Importing Data 3/15 2/3')
train_['base_lgb'] = train_base_lgb['base_lgb']
print_step('Importing Data 3/15 3/3')
test_['base_lgb'] = test_base_lgb['base_lgb']

print_step('Importing Data 3/15 1/3')
train_base_lgb, test_base_lgb = load_cache('base_lgb2')
print_step('Importing Data 3/15 2/3')
train_['base_lgb2'] = train_base_lgb['base_lgb2']
print_step('Importing Data 3/15 3/3')
test_['base_lgb2'] = test_base_lgb['base_lgb2']

train_te_lgb, test_te_lgb = load_cache('te_lgb')
print_step('Importing Data 3/15 2/3')
train_['te_lgb'] = train_te_lgb['te_lgb']
print_step('Importing Data 3/15 3/3')
test_['te_lgb'] = test_te_lgb['te_lgb']

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb2')
print_step('Importing Data 3/15 2/3')
train_['te_lgb2'] = train_te_lgb2['te_lgb']
print_step('Importing Data 3/15 3/3')
test_['te_lgb2'] = test_te_lgb2['te_lgb']

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb3')
print_step('Importing Data 3/15 2/3')
train_['te_lgb3'] = train_te_lgb2['te_lgb3']
print_step('Importing Data 3/15 3/3')
test_['te_lgb3'] = test_te_lgb2['te_lgb3']

print_step('Importing Data 3/15 1/3')
train_base_lgb2, test_base_lgb2 = load_cache('base_lgb_poisson')
print_step('Importing Data 3/15 2/3')
train_['base_lgb_poisson'] = train_base_lgb2['base_lgb_poisson']
print_step('Importing Data 3/15 3/3')
test_['base_lgb_poisson'] = test_base_lgb2['te_lgb_poisson'] # typo

print_step('Importing Data 3/15 1/3')
train_te_lgb2, test_te_lgb2 = load_cache('te_lgb_poisson')
print_step('Importing Data 3/15 2/3')
train_['te_lgb_poisson'] = train_te_lgb2['te_lgb_poisson']
print_step('Importing Data 3/15 3/3')
test_['te_lgb_poisson'] = test_te_lgb2['te_lgb_poisson']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v29')
print_step('Importing Data 3/15 2/3')
train_['ryan_lgbm_v29'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_['ryan_lgbm_v29'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v33')
print_step('Importing Data 3/15 2/3')
train_['ryan_lgbm_v33'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_['ryan_lgbm_v33'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 3/15 1/3')
train_ryan_lgbm_v29, test_ryan_lgbm_v29 = load_cache('ryan_lgbm_v36')
print_step('Importing Data 3/15 2/3')
train_['ryan_lgbm_v36'] = train_ryan_lgbm_v29['oof_lgbm']
print_step('Importing Data 3/15 3/3')
test_['ryan_lgbm_v36'] = test_ryan_lgbm_v29['oof_lgbm']

print_step('Importing Data 3/15 1/3')
train_ridge_lgb, test_ridge_lgb = load_cache('ridge_lgb')
print_step('Importing Data 3/15 2/3')
train_['ridge_lgb'] = train_ridge_lgb['ridge_lgb']
print_step('Importing Data 3/15 3/3')
test_['ridge_lgb'] = test_ridge_lgb['ridge_lgb']

print_step('Importing Data 3/15 1/3')
train_ridge_lgb, test_ridge_lgb = load_cache('ridge_lgb2')
print_step('Importing Data 3/15 2/3')
train_['ridge_lgb2'] = train_ridge_lgb['ridge_lgb2']
print_step('Importing Data 3/15 3/3')
test_['ridge_lgb2'] = test_ridge_lgb['ridge_lgb2']

print_step('Importing Data 3/15 1/3')
train_ridge_lgb, test_ridge_lgb = load_cache('ridge_lgb3')
print_step('Importing Data 3/15 2/3')
train_['ridge_lgb3'] = train_ridge_lgb['ridge_lgb3']
print_step('Importing Data 3/15 3/3')
test_['ridge_lgb3'] = test_ridge_lgb['ridge_lgb3']

print_step('Importing Data 3/15 1/3')
train_ridge_lgb, test_ridge_lgb = load_cache('ridge_lgb_poisson')
print_step('Importing Data 3/15 2/3')
train_['ridge_lgb_poisson'] = train_ridge_lgb['ridge_lgb_poisson']
print_step('Importing Data 3/15 3/3')
test_['ridge_lgb_poisson'] = test_ridge_lgb['ridge_lgb_poisson']

print_step('Importing Data 4/15 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 4/15 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 4/15 3/4')
train_ = pd.concat([train_, train_pcat_ridge], axis=1)
print_step('Importing Data 4/15 4/4')
test_ = pd.concat([test_, test_pcat_ridge], axis=1)

print_step('Importing Data 5/15 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 5/15 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 5/15 3/4')
train_ = pd.concat([train_, train_rcat_ridge], axis=1)
print_step('Importing Data 5/15 4/4')
test_ = pd.concat([test_, test_rcat_ridge], axis=1)

print_step('Importing Data 6/15 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 6/15 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 6/15 3/4')
train_ = pd.concat([train_, train_catb_ridge], axis=1)
print_step('Importing Data 6/15 4/4')
test_ = pd.concat([test_, test_catb_ridge], axis=1)

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 7/15 2/3')
train_['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 7/15 3/3')
test_['deep_lgb'] = test_deep_lgb['deep_lgb']

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb2')
print_step('Importing Data 7/15 2/3')
train_['deep_lgb2'] = train_deep_lgb['deep_lgb2']
print_step('Importing Data 7/15 3/3')
test_['deep_lgb2'] = test_deep_lgb['deep_lgb2']

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb3')
print_step('Importing Data 7/15 2/3')
train_['deep_lgb3'] = train_deep_lgb['deep_lgb3']
print_step('Importing Data 7/15 3/3')
test_['deep_lgb3'] = test_deep_lgb['deep_lgb3']

print_step('Importing Data 7/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb4')
print_step('Importing Data 7/15 2/3')
train_['deep_lgb4'] = train_deep_lgb['deep_lgb4']
print_step('Importing Data 7/15 3/3')
test_['deep_lgb4'] = test_deep_lgb['deep_lgb4']

print_step('Importing Data 8/15 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 8/15 2/3')
train_['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 8/15 3/3')
test_['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 9/15 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 9/15 2/3')
train_['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 9/15 3/3')
test_['complete_ridge'] = test_complete_ridge['complete_ridge']

print_step('Importing Data 9/15 1/3')
train_ryan_ridge, test_ryan_ridge = load_cache('ryan_ridge_sgd_v2')
print_step('Importing Data 9/15 2/3')
train_['ryan_ridge'] = train_ryan_ridge['oof_ridge']
print_step('Importing Data 9/15 3/3')
test_['ryan_ridge'] = test_ryan_ridge['oof_ridge']
print_step('Importing Data 9/15 2/3')
train_['ryan_sgd'] = train_ryan_ridge['oof_sgd']
print_step('Importing Data 9/15 3/3')
test_['ryan_sgd'] = test_ryan_ridge['oof_sgd']

print_step('Importing Data 10/15 1/3')
train_complete_fm, test_complete_fm = load_cache('complete_fm')
print_step('Importing Data 10/15 2/3')
train_['complete_fm'] = train_complete_fm['complete_fm']
print_step('Importing Data 10/15 3/3')
test_['complete_fm'] = test_complete_fm['complete_fm']

print_step('Importing Data 11/15 1/3')
train_tffm2, test_tffm2 = load_cache('tffm2')
print_step('Importing Data 11/15 2/3')
train_['tffm2'] = train_tffm2['tffm2']
print_step('Importing Data 11/15 3/3')
test_['tffm2'] = test_tffm2['tffm2']

print_step('Importing Data 12/15 1/3')
train_tffm3, test_tffm3 = load_cache('tffm3')
print_step('Importing Data 12/15 2/3')
train_['tffm3'] = train_tffm3['tffm3']
print_step('Importing Data 12/15 3/3')
test_['tffm3'] = test_tffm3['tffm3']

print_step('Importing Data 13/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_FastText')
print_step('Importing Data 13/15 2/3')
train_['cnn_ft'] = train_cnn_ft['CNN_FastText']
print_step('Importing Data 13/15 3/3')
test_['cnn_ft'] = test_cnn_ft['CNN_FastText']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_FastText_4')
print_step('Importing Data 14/15 2/3')
train_['cnn_ft4'] = train_cnn_ft['CNN_FastText_4']
print_step('Importing Data 14/15 3/3')
test_['cnn_ft4'] = test_cnn_ft['CNN_FastText_4']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('RNN_AttentionPooling')
print_step('Importing Data 14/15 2/3')
train_['rnn_at'] = train_cnn_ft['RNN_AttentionPooling']
print_step('Importing Data 14/15 3/3')
test_['rnn_at'] = test_cnn_ft['RNN_AttentionPooling']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('RNN_AttentionPooling_img2')
print_step('Importing Data 14/15 2/3')
train_['rnn_at2'] = train_cnn_ft['RNN_AttentionPooling_img2']
print_step('Importing Data 14/15 3/3')
test_['rnn_at2'] = test_cnn_ft['RNN_AttentionPooling_img2']

print_step('Importing Data 14/15 1/3')
train_cnn_ft = pd.read_csv('cache/matt_nn_oof.csv')
test_cnn_ft = pd.read_csv('cache/matt_nn_test.csv')
print_step('Importing Data 14/15 2/3')
train_['matt_nn'] = train_cnn_ft['matt_nn']
print_step('Importing Data 14/15 3/3')
test_['matt_nn'] = test_cnn_ft['deal_probability']

train_multi = pd.read_csv('cache/matt_multi_6_oof.csv')
test_multi = pd.read_csv('cache/matt_multi_6_test.csv')
train_multi.drop('item_id', axis=1, inplace=True)
test_multi.drop('item_id', axis=1, inplace=True)
train_multi.columns = ['matt_6_' + c for c in train_multi.columns]
test_multi.columns = ['matt_6_' + c for c in test_multi.columns]
print_step('Importing Data 14/15 2/3')
train_ = pd.concat([train_, train_multi], axis=1)
print_step('Importing Data 14/15 3/3')
test_ = pd.concat([test_, test_multi], axis=1)

print_step('Importing Data 14/15 1/3')
train_multi = pd.read_csv('cache/matt_multi_5_oof.csv')
test_multi = pd.read_csv('cache/matt_multi_5_test.csv')
train_multi.drop('item_id', axis=1, inplace=True)
test_multi.drop('item_id', axis=1, inplace=True)
train_multi.columns = ['matt_5_' + c for c in train_multi.columns]
test_multi.columns = ['matt_5_' + c for c in test_multi.columns]
print_step('Importing Data 14/15 2/3')
train_ = pd.concat([train_, train_multi], axis=1)
print_step('Importing Data 14/15 3/3')
test_ = pd.concat([test_, test_multi], axis=1)

print_step('Importing Data 14/15 1/3')
train_multi = pd.read_csv('cache/train_liu_nn_multiclass.csv')
test_multi = pd.read_csv('cache/test_liu_nn_multiclass.csv')
train_multi.columns = ['liu_' + c for c in train_multi.columns]
test_multi.columns = ['liu_' + c for c in test_multi.columns]
print_step('Importing Data 14/15 2/3')
train_ = pd.concat([train_, train_multi], axis=1)
print_step('Importing Data 14/15 3/3')
test_ = pd.concat([test_, test_multi], axis=1)
import pdb
pdb.set_trace()

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_binary')
print_step('Importing Data 14/15 2/3')
train_['CNN_binary'] = train_cnn_ft['CNN_binary']
print_step('Importing Data 14/15 3/3')
test_['CNN_binary'] = test_cnn_ft['CNN_binary']

print_step('Importing Data 14/15 1/3')
train_cnn_ft, test_cnn_ft = load_cache('CNN_binary_PL')
print_step('Importing Data 14/15 2/3')
train_['CNN_binary_PL'] = train_cnn_ft['CNN_binary_PL']
print_step('Importing Data 14/15 3/3')
test_['CNN_binary_PL'] = test_cnn_ft['CNN_binary_PL']

print_step('Importing Data 14/15 1/3')
train_liu_nn, test_liu_nn = load_cache('liu_nn')
print_step('Importing Data 14/15 2/3')
train_['liu_nn'] = train_liu_nn['liu_nn']
print_step('Importing Data 14/15 3/3')
test_['liu_nn'] = test_liu_nn['liu_nn']

print_step('Importing Data 14/15 1/3')
train_liu_nn, test_liu_nn = load_cache('liu_nn2')
print_step('Importing Data 14/15 2/3')
train_['liu_nn2'] = train_liu_nn['liu_nn2']
print_step('Importing Data 14/15 3/3')
test_['liu_nn2'] = test_liu_nn['liu_nn2']

print_step('Importing Data 14/15 1/3')
train_liu_lgb, test_liu_lgb = load_cache('liu_lgb')
print_step('Importing Data 14/15 2/3')
train_['liu_lgb'] = train_liu_lgb['liu_lgb']
print_step('Importing Data 14/15 3/3')
test_['liu_lgb'] = test_liu_lgb['liu_lgb']


models = [c for c in train_.columns if 'svd' not in c and 'price' not in c and 'img' not in c and 'parent_category' not in c]
pprint(sorted([(m, rmse(target, train_[m])) for m in models], key = lambda x: x[1]))
good_models = [m for m in models if 'lgb' in m or 'nn' in m or 'NN' in m or 'fm' in m]
pd.set_option('display.max_columns', 500)
print(pd.DataFrame(np.corrcoef([train_[m] for m in good_models]), index = good_models, columns = good_models))

print_step('Importing Data 15/15 1/4')
train_img, test_img = load_cache('img_data')
cols = ['img_dullness_light_percent', 'img_dullness_dark_percent']
train_img = train_img[cols].fillna(0)
test_img = test_img[cols].fillna(0)
print_step('Importing Data 15/15 2/4')
train_fe = pd.concat([train_fe, train_img], axis=1)
print_step('Importing Data 15/15 3/4')
test_fe = pd.concat([test_fe, test_img], axis=1)
print_step('Importing Data 15/15 4/4')
train_['img_dullness_light_percent'] = train_fe['img_dullness_light_percent']
test_['img_dullness_light_percent'] = test_fe['img_dullness_light_percent']
train_['img_dullness_dark_percent'] = train_fe['img_dullness_dark_percent']
test_['img_dullness_dark_percent'] = test_fe['img_dullness_dark_percent']

print_step('Importing Data 15/15 4/4')
train_embeddings_df, test_embeddings_df = load_cache('avito_fasttext_300d')
print_step('Embedding SVD 1/4')
NCOMP = 20
svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
svd.fit(train_embeddings_df)
print_step('Embedding SVD 2/4')
train_svd = pd.DataFrame(svd.transform(train_embeddings_df))
print_step('Embedding SVD 3/4')
test_svd = pd.DataFrame(svd.transform(test_embeddings_df))
print_step('Embedding SVD 4/4')
train_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
test_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
train_ = pd.concat([train_, train_svd], axis=1)
test_ = pd.concat([test_, test_svd], axis=1)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'day_of_week']
for col in train_.columns:
    print(col)
    if col in cat_cols:
        train_[col] = train_[col].astype('category')
        test_[col] = test_[col].astype('category')
    else:
        train_[col] = train_[col].astype(np.float64)
        test_[col] = test_[col].astype(np.float64)

print('~~~~~~~~~~~~')
print_step('Run LGB')
print(train_.shape)
print(test_.shape)
results = run_cv_model(train_, test_, target, runLGB, params, rmse, 'lgb_blender')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lgb_blender', pd.DataFrame({'lgb_blender': results['train']}),
                             pd.DataFrame({'lgb_blender': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb_blender.csv', index=False)
print_step('Done!')

print('~~~~~~~~~~~~~~~~~~~~')
print_step('Run Poisson LGB')
print(train_.shape)
print(test_.shape)
poisson_results = run_cv_model(train_, test_, target, runLGB, poisson_params, rmse, 'possion_lgb_blender')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('lgb_blender_poisson', pd.DataFrame({'lgb_blender_poisson': poisson_results['train']}),
                                     pd.DataFrame({'lgb_blender_poisson': poisson_results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = poisson_results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb_blender_poisson.csv', index=False)
print_step('Done!')

print('~~~~~~~~~~~~~~~~')
print_step('Run Average')
average_results_train = results['train'].clip(0.0, 1.0) * 0.5 + poisson_results['train'].clip(0.0, 1.0) * 0.5
average_results_test = results['test'].clip(0.0, 1.0) * 0.5 + poisson_results['test'].clip(0.0, 1.0) * 0.5
print('RMSE: ' + str(rmse(target, average_results_train)))
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('blender_average', pd.DataFrame({'blender_average': average_results_train}),
                                 pd.DataFrame({'blender_average': average_results_test}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = average_results_test.clip(0.0, 1.0)
submission.to_csv('submit/submit_average_blender.csv', index=False)
print_step('Done!')
