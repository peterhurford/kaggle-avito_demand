import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import Lasso

from cv import run_cv_model
from utils import rmse, normalize_text, print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


params = {
    "alpha": 1e-5,
    "max_iter": 1500,
    "positive": True
}
def runLasso(train_X, train_y, val_X, val_y, test_X, params):
    model = Lasso(**params)
    model.fit(train_X, train_y)
    for i in zip(train_X.columns, model.coef_):
        print(i)
    print_step('Predict Val 1/2')
    pred_val_y = model.predict(val_X)
    print_step('Predict Test 2/2')
    pred_test_y = model.predict(test_X)
    return pred_val_y, pred_test_y


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()
target = train['deal_probability']
test_id = test['item_id']

print_step('Importing Data 2/15')
train_ridge, test_ridge = load_cache('tfidf_ridges')
drops = [c for c in train_ridge.columns if 'svd' in c or 'tfidf' in c]
train_ridge.drop(drops, axis=1, inplace=True)
test_ridge.drop(drops, axis=1, inplace=True)
train_ = train_ridge
test_ = test_ridge

print_step('Importing Data 1/5')
train_['deal_probability'] = target
train_['item_id'] = train['item_id']
test_['item_id'] = test['item_id']

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

def run_ridge_on_cat(cat):
    if not is_in_cache('cat_ridges_blend_l2_' + cat):
        print_step(cat + ' > Subsetting')
        train_c = train_[train['parent_category_name'] == cat].copy()
        test_c = test_[test['parent_category_name'] == cat].copy()
        print(train_c.shape)
        print(test_c.shape)
        target = train_c['deal_probability'].values
        train_id = train_c['item_id']
        test_id = test_c['item_id']
        train_c.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
        test_c.drop('item_id', axis=1, inplace=True)

        print_step(cat + ' > Modeling')
        results = run_cv_model(train_c, test_c, target, runLasso, params, rmse, cat + '-ridge-blend')
        train_c['cat_ridge'] = results['train']
        test_c['cat_ridge'] = results['test']
        print_step(cat + ' > RMSE: ' + str(rmse(target, train_c['cat_ridge'])))

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print_step(cat + ' > Saving in Cache')
        train_c['item_id'] = train_id
        test_c['item_id'] = test_id
        save_in_cache('cat_ridges_blend_l2_' + cat, train_c[['item_id', 'cat_ridge']], test_c[['item_id', 'cat_ridge']])
        return True
    else:
        print_step('Already have ' + cat + '...')
        return True

print_step('Compiling set')
parent_cats = train['parent_category_name'].unique()
n_cpu = mp.cpu_count()
n_nodes = 14
print_step('Starting a jobs server with %d nodes' % n_nodes)
pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
res = pool.map(run_ridge_on_cat, parent_cats)
pool.close()
pool.join()
pool.terminate()
pool.restart()

print('~~~~~~~~~~~~~~~~')
print_step('Merging 1/5')
pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
dfs = pool.map(lambda c: load_cache('cat_ridges_blend_l2_' + c), parent_cats)
pool.close()
pool.join()
pool.terminate()
pool.restart()
print_step('Merging 2/5')
train_dfs = map(lambda x: x[0], dfs)
test_dfs = map(lambda x: x[1], dfs)
print_step('Merging 3/5')
train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)
print_step('Merging 4/5')
train_lasso = train.merge(train_df, on='item_id')
print_step('Merging 5/5')
test_lasso = test.merge(test_df, on='item_id')
print_step('RMSE: ' + str(rmse(train_lasso['deal_probability'], train_lasso['cat_ridge'])))
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('pc_lasso_l2', pd.DataFrame({'pc_lasso_l2': train_lasso['cat_ridge']}),
                             pd.DataFrame({'pc_lasso_l2': test_lasso['cat_ridge']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = test_lasso['cat_ridge'].clip(0.0, 1.0)
submission.to_csv('submit/submit_pc_lasso_l2_blender.csv', index=False)
print_step('Done!')
