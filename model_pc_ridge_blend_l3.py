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
print_step('Importing Data 1/5')
train_, test_ = load_cache('lgb_blender')
train_['deal_probability'] = target
train_['item_id'] = train['item_id']
test_['item_id'] = test['item_id']

print_step('Importing Data 2/5')
train_te_lgb2, test_te_lgb2 = load_cache('lgb_blender_poisson')
train_['lgb_blender_poisson'] = train_te_lgb2['lgb_blender_poisson']
test_['lgb_blender_poisson'] = test_te_lgb2['lgb_blender_poisson']

print_step('Importing Data 3/5')
train_te_lgb2, test_te_lgb2 = load_cache('flat_blender_lgb')
train_['flat_blender_lgb'] = train_te_lgb2['flat_blender_lgb']
test_['flat_blender_lgb'] = test_te_lgb2['flat_blender_lgb']

print_step('Importing Data 4/5')
train_te_lgb2, test_te_lgb2 = load_cache('MLP_blender')
train_['MLP_blender'] = train_te_lgb2['MLP_blender']
test_['MLP_blender'] = test_te_lgb2['MLP_blender']

print_step('Importing Data 5/5')
train_te_lgb2, test_te_lgb2 = load_cache('lasso_blender')
train_['lasso_blender'] = train_te_lgb2['lasso_blender']
test_['lasso_blender'] = test_te_lgb2['lasso_blender']

print_step('Importing Data 5/5')
train_te_lgb2, test_te_lgb2 = load_cache('pc_lasso_l2')
train_['pc_lasso_l2'] = train_te_lgb2['pc_lasso_l2']
test_['pc_lasso_l2'] = test_te_lgb2['pc_lasso_l2']

def run_ridge_on_cat(cat):
    if not is_in_cache('cat_ridges_blend_l3_' + cat):
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
        save_in_cache('cat_ridges_blend_l3_' + cat, train_c[['item_id', 'cat_ridge']], test_c[['item_id', 'cat_ridge']])
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
dfs = pool.map(lambda c: load_cache('cat_ridges_blend_l3_' + c), parent_cats)
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
save_in_cache('pc_lasso_l3', pd.DataFrame({'pc_lasso_l3': train_lasso['cat_ridge']}),
                             pd.DataFrame({'pc_lasso_l3': test_lasso['cat_ridge']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_pc_lasso_l3_blender.csv', index=False)
print_step('Done!')
