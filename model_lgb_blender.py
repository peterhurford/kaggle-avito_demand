from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runLGB(train_X, train_y, test_X, test_y, test_X2):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.02,
              'application': 'regression',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.8,
              'nthread': 16, #max(mp.cpu_count() - 2, 2),
              'lambda_l1': 1,
              'lambda_l2': 1,
              'min_data_in_leaf': 40}
    print_step('Train LGB')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=400,
                      valid_sets=watchlist,
                      verbose_eval=100)
    print_step('Feature importance')
    pprint(sorted(list(zip(model.feature_importance(), train_X.columns)), reverse=True))
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/14')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 2/14 2/2')
train_fe = train_ridge
test_fe = test_ridge

train_fe['parent_category_name'] = train['parent_category_name']
test_fe['parent_category_name'] = test['parent_category_name']
train_fe['price'] = train['price']
test_fe['price'] = test['price']

print_step('Importing Data 3/14 1/3')
train_ridge_lgb, test_ridge_lgb = load_cache('ridge_lgb')
print_step('Importing Data 3/14 2/3')
train_fe['ridge_lgb'] = train_ridge_lgb['ridge_lgb']
print_step('Importing Data 3/14 3/3')
test_fe['ridge_lgb'] = test_ridge_lgb['ridge_lgb']

print_step('Importing Data 3/14 1/3')
train_stack_lgb, test_stack_lgb = load_cache('stack_lgb')
print_step('Importing Data 3/14 2/3')
train_fe['stack_lgb'] = train_stack_lgb['stack_lgb']
print_step('Importing Data 3/14 3/3')
test_fe['stack_lgb'] = test_stack_lgb['stack_lgb']

print_step('Importing Data 6/12 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 6/12 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 6/12 3/4')
train_fe = pd.concat([train_fe, train_pcat_ridge], axis=1)
print_step('Importing Data 6/12 4/4')
test_fe = pd.concat([test_fe, test_pcat_ridge], axis=1)

print_step('Importing Data 7/12 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 7/12 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 7/12 3/4')
train_fe = pd.concat([train_fe, train_rcat_ridge], axis=1)
print_step('Importing Data 7/12 4/4')
test_fe = pd.concat([test_fe, test_rcat_ridge], axis=1)

print_step('Importing Data 8/12 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 8/12 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 8/12 3/4')
train_fe = pd.concat([train_fe, train_catb_ridge], axis=1)
print_step('Importing Data 8/12 4/4')
test_fe = pd.concat([test_fe, test_catb_ridge], axis=1)

print_step('Importing Data 6/14 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 6/14 2/3')
train_fe['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 6/14 3/3')
test_fe['deep_lgb'] = test_deep_lgb['deep_lgb']

print_step('Importing Data 7/14 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 7/14 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 7/14 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 8/14 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 8/14 2/3')
train_fe['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 8/14 3/3')
test_fe['complete_ridge'] = test_complete_ridge['complete_ridge']

print_step('Importing Data 9/14 1/3')
train_complete_fm, test_complete_fm = load_cache('complete_fm')
print_step('Importing Data 9/14 2/3')
train_fe['complete_fm'] = train_complete_fm['complete_fm']
print_step('Importing Data 9/14 3/3')
test_fe['complete_fm'] = test_complete_fm['complete_fm']

print_step('Importing Data 10/14 1/3')
train_tffm2, test_tffm2 = load_cache('tffm2')
print_step('Importing Data 10/14 2/3')
train_fe['tffm2'] = train_tffm2['tffm2']
print_step('Importing Data 10/14 3/3')
test_fe['tffm2'] = test_tffm2['tffm2']

print_step('Importing Data 11/14 1/3')
train_tffm3, test_tffm3 = load_cache('tffm3')
print_step('Importing Data 11/14 2/3')
train_fe['tffm3'] = train_tffm3['tffm3']
print_step('Importing Data 11/14 3/3')
test_fe['tffm3'] = test_tffm3['tffm3']

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'day_of_week']
for col in train_fe.columns:
    print(col)
    if col in cat_cols:
        train_fe[col] = train_fe[col].astype('category')
        test_fe[col] = test_fe[col].astype('category')
    else:
        train_fe[col] = train_fe[col].astype(np.float64)
        test_fe[col] = test_fe[col].astype(np.float64)

print(train_fe.columns)
print(train_fe.shape)
print(test_fe.shape)

print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'lgb_blender')
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

# [2018-06-12 03:52:21.398025] lgb_blender cv scores : [0.21405754457786552, 0.21318059827171135, 0.2130803173508629, 0.21308862016317834, 0.21370161844415586]
# [2018-06-12 03:52:21.398105] lgb_blender mean cv score : 0.21342173976155476
# [2018-06-12 03:52:21.398209] lgb_blender std cv score : 0.000391986476523742

# [100]   training's rmse: 0.214269       valid_1's rmse: 0.215193
# [200]   training's rmse: 0.212898       valid_1's rmse: 0.2141
# [300]   training's rmse: 0.212602       valid_1's rmse: 0.214056
# [400]   training's rmse: 0.212378       valid_1's rmse: 0.214053
