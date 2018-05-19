from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache


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
              'nthread': max(mp.cpu_count() - 2, 2),
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


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/8')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/8')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 3/8')
train_ridge, test_ridge = load_cache('tfidf_ridges')

print_step('Importing Data 4/8')
train_fe = pd.concat([train_fe, train_ridge], axis=1)
test_fe = pd.concat([test_fe, test_ridge], axis=1)

print_step('Importing Data 5/8')
train_deep_text_lgb, test_deep_text_lgb = load_cache('deep_text_lgb')

print_step('Importing Data 6/8')
train_fe['deep_text_lgb'] = train_deep_text_lgb['deep_text_lgb']
test_fe['deep_text_lgb'] = test_deep_text_lgb['deep_text_lgb']

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 7/8')
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 8/8')
region_macro = pd.read_csv('region_macro.csv')
train_fe = train_fe.merge(region_macro, how='left', on='region')
test_fe = test_fe.merge(region_macro, how='left', on='region')

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
save_in_cache('lgb_preds8', pd.DataFrame({'lgb': results['train']}),
                            pd.DataFrame({'lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb8.csv', index=False)
print_step('Done!')

# CURRENT
# [2018-05-19 20:22:03.391174] lgb cv scores : [0.217888591043512, 0.21691784888621873, 0.21712214418861964, 0.21678523082466095, 0.21742469115264887]
# [2018-05-19 20:22:03.391241] lgb mean cv score : 0.21722770121913207
# [2018-05-19 20:22:03.391344] lgb std cv score : 0.0003945912312157352

# [100]   training's rmse: 0.216548       valid_1's rmse: 0.220377
# [200]   training's rmse: 0.212651       valid_1's rmse: 0.218937
# [300]   training's rmse: 0.210151       valid_1's rmse: 0.218478
# [400]   training's rmse: 0.208163       valid_1's rmse: 0.218246
# [500]   training's rmse: 0.20651        valid_1's rmse: 0.218126
# [600]   training's rmse: 0.205002       valid_1's rmse: 0.218041
# [700]   training's rmse: 0.203739       valid_1's rmse: 0.217995
# [800]   training's rmse: 0.202489       valid_1's rmse: 0.217945
# [900]   training's rmse: 0.201415       valid_1's rmse: 0.217894
# [1000]  training's rmse: 0.200331       valid_1's rmse: 0.217889
