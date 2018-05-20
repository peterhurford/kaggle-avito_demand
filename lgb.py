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
    params = {'learning_rate': 0.03,
              'application': 'regression',
              'num_leaves': 118,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.15,
              'nthread': max(mp.cpu_count() - 2, 2),
              'lambda_l1': 6,
              'lambda_l2': 6,
              'min_data_in_leaf': 40}
    print_step('Train LGB')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=1800,
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
print_step('Importing Data 1/11')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/11')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 3/11 1/3')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 3/11 2/3')
train_fe = pd.concat([train_fe, train_ridge], axis=1)
print_step('Importing Data 3/11 3/3')
test_fe = pd.concat([test_fe, test_ridge], axis=1)

print_step('Importing Data 4/11 1/3')
train_deep_text_lgb, test_deep_text_lgb = load_cache('deep_text_lgb')
print_step('Importing Data 4/11 2/3')
train_fe['deep_text_lgb'] = train_deep_text_lgb['deep_text_lgb']
print_step('Importing Data 4/11 3/3')
test_fe['deep_text_lgb'] = test_deep_text_lgb['deep_text_lgb']

print_step('Importing Data 5/11 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 5/11 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 5/11 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 6/11 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 6/11 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 6/11 3/4')
train_fe = pd.concat([train_fe, train_pcat_ridge], axis=1)
print_step('Importing Data 6/11 4/4')
test_fe = pd.concat([test_fe, test_pcat_ridge], axis=1)

print_step('Importing Data 7/11 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 7/11 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 7/11 3/4')
train_fe = pd.concat([train_fe, train_rcat_ridge], axis=1)
print_step('Importing Data 7/11 4/4')
test_fe = pd.concat([test_fe, test_rcat_ridge], axis=1)

print_step('Importing Data 8/11 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 8/11 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 8/11 3/4')
train_fe = pd.concat([train_fe, train_catb_ridge], axis=1)
print_step('Importing Data 8/11 4/4')
test_fe = pd.concat([test_fe, test_catb_ridge], axis=1)

# print_step('Importing Data 9/11 1/4')
# train_region_ridge, test_region_ridge = load_cache('region_bin_ridges')
# print_step('Importing Data 9/11 2/4')
# train_region_ridge = train_region_ridge[[c for c in train_region_ridge.columns if 'ridge' in c]]
# test_region_ridge = test_region_ridge[[c for c in test_region_ridge.columns if 'ridge' in c]]
# print_step('Importing Data 9/11 3/4')
# train_fe = pd.concat([train_fe, train_region_ridge], axis=1)
# print_step('Importing Data 9/11 4/4')
# test_fe = pd.concat([test_fe, test_region_ridge], axis=1)

print_step('Importing Data 10/11 1/4')
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
print_step('Importing Data 10/11 2/4')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 10/11 3/4')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 10/11 4/4')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

print_step('Importing Data 11/11 1/3')
region_macro = pd.read_csv('region_macro.csv')
print_step('Importing Data 11/11 2/3')
train_fe = train_fe.merge(region_macro, how='left', on='region')
print_step('Importing Data 11/11 3/3')
test_fe = test_fe.merge(region_macro, how='left', on='region')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
train_fe['image_top_1'] = train_fe['image_top_1'].astype('str').fillna('missing')
test_fe['image_top_1'] = test_fe['image_top_1'].astype('str').fillna('missing')
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
save_in_cache('lgb_preds10', pd.DataFrame({'lgb': results['train']}),
                             pd.DataFrame({'lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb10.csv', index=False)
print_step('Done!')

# CURRENT
# [2018-05-20 07:46:21.375440] lgb cv scores : [0.21678302448776876, 0.21572820645250634, 0.21590908088669739, 0.21578072469266416, 0.21636757405468007]
# [2018-05-20 07:46:21.375506] lgb mean cv score : 0.21611372211486332
# [2018-05-20 07:46:21.375613] lgb std cv score : 0.0004034337901328224

# [100]   training's rmse: 0.217606       valid_1's rmse: 0.220554
# [200]   training's rmse: 0.214243       valid_1's rmse: 0.218804
# [300]   training's rmse: 0.212097       valid_1's rmse: 0.218026
# [400]   training's rmse: 0.210507       valid_1's rmse: 0.21769
# [500]   training's rmse: 0.209217       valid_1's rmse: 0.217435
# [600]   training's rmse: 0.207899       valid_1's rmse: 0.217276
# [700]   training's rmse: 0.206744       valid_1's rmse: 0.21717
# [800]   training's rmse: 0.205775       valid_1's rmse: 0.217068
# [900]   training's rmse: 0.204834       valid_1's rmse: 0.217011
# [1000]  training's rmse: 0.203854       valid_1's rmse: 0.216973
# [1100]  training's rmse: 0.203018       valid_1's rmse: 0.216943
# [1200]  training's rmse: 0.202226       valid_1's rmse: 0.216895
# [1300]  training's rmse: 0.201483       valid_1's rmse: 0.216857
# [1400]  training's rmse: 0.200783       valid_1's rmse: 0.216842
# [1500]  training's rmse: 0.200083       valid_1's rmse: 0.216829
# [1600]  training's rmse: 0.199426       valid_1's rmse: 0.216807
# [1700]  training's rmse: 0.198804       valid_1's rmse: 0.216793
# [1800]  training's rmse: 0.198208       valid_1's rmse: 0.216783
