from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.decomposition import TruncatedSVD

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
              'num_leaves': 250,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.1, # 0.15
              'nthread': 16, #max(mp.cpu_count() - 2, 2),
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
print_step('Importing Data 1/12')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/12')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 3/12 1/3')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 3/12 2/3')
train_fe = pd.concat([train_fe, train_ridge], axis=1)
print_step('Importing Data 3/12 3/3')
test_fe = pd.concat([test_fe, test_ridge], axis=1)

print_step('Importing Data 4/12 1/3')
train_deep_text_lgb, test_deep_text_lgb = load_cache('deep_text_lgb')
print_step('Importing Data 4/12 2/3')
train_fe['deep_text_lgb'] = train_deep_text_lgb['deep_text_lgb']
print_step('Importing Data 4/12 3/3')
test_fe['deep_text_lgb'] = test_deep_text_lgb['deep_text_lgb']

print_step('Importing Data 5/12 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 5/12 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 5/12 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

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

print_step('Importing Data 9/12 1/4')
train_img, test_img = load_cache('img_data')
print_step('Importing Data 9/12 2/4')
cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
train_hist = train_img[[c for c in train_img.columns if 'histogram' in c]].fillna(0)
test_hist = test_img[[c for c in test_img.columns if 'histogram' in c]].fillna(0)
train_img = train_img[cols].fillna(0)
test_img = test_img[cols].fillna(0)
print_step('Importing Data 9/12 3/4')
train_fe = pd.concat([train_fe, train_img], axis=1)
print_step('Importing Data 9/12 4/4')
test_fe = pd.concat([test_fe, test_img], axis=1)

print_step('Importing Data 10/12 1/4')
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
print_step('Importing Data 10/12 2/4')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 10/12 3/4')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 10/12 4/4')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

print_step('Importing Data 11/12 1/3')
region_macro = pd.read_csv('region_macro.csv')
print_step('Importing Data 11/12 2/3')
train_fe = train_fe.merge(region_macro, how='left', on='region')
print_step('Importing Data 11/12 3/3')
test_fe = test_fe.merge(region_macro, how='left', on='region')

print_step('Importing Data 12/12 1/4')
train_active_feats, test_active_feats = load_cache('active_feats')
print_step('Importing Data 12/12 2/4')
train_active_feats.drop('user_id', axis=1, inplace=True)
test_active_feats.drop('user_id', axis=1, inplace=True)
print_step('Importing Data 12/12 3/4')
train_fe = pd.concat([train_fe, train_active_feats], axis=1)
print_step('Importing Data 12/12 3/4')
test_fe = pd.concat([test_fe, test_active_feats], axis=1)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
train_fe['image_top_1'] = train_fe['image_top_1'].astype('str').fillna('missing')
test_fe['image_top_1'] = test_fe['image_top_1'].astype('str').fillna('missing')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'day_of_week',
            'img_average_color']
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
save_in_cache('lgb_preds12', pd.DataFrame({'lgb': results['train']}),
                             pd.DataFrame({'lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb12.csv', index=False)
print_step('Done!')

# CURRENT
# [2018-05-28 02:26:09.920393] lgb cv scores : [0.21469048839471144, 0.21380327230845902, 0.2138388386785356, 0.2137308459589358, 0.21429875684372357]
# [2018-05-28 02:26:09.920463] lgb mean cv score : 0.2140724404368731
# [2018-05-28 02:26:09.920567] lgb std cv score : 0.0003679430511390113

# [100]   training's rmse: 0.216676       valid_1's rmse: 0.220363
# [200]   training's rmse: 0.211711       valid_1's rmse: 0.217885
# [300]   training's rmse: 0.208408       valid_1's rmse: 0.216851
# [400]   training's rmse: 0.205916       valid_1's rmse: 0.216251
# [500]   training's rmse: 0.203571       valid_1's rmse: 0.215851
# [600]   training's rmse: 0.201495       valid_1's rmse: 0.215541
# [700]   training's rmse: 0.199626       valid_1's rmse: 0.21533
# [800]   training's rmse: 0.197779       valid_1's rmse: 0.215173
# [900]   training's rmse: 0.19621        valid_1's rmse: 0.215069
# [1000]  training's rmse: 0.19467        valid_1's rmse: 0.214991
# [1100]  training's rmse: 0.193204       valid_1's rmse: 0.214914
# [1200]  training's rmse: 0.191844       valid_1's rmse: 0.214879
# [1300]  training's rmse: 0.190496       valid_1's rmse: 0.214832
# [1400]  training's rmse: 0.189183       valid_1's rmse: 0.214819
# [1500]  training's rmse: 0.187879       valid_1's rmse: 0.214757
# [1600]  training's rmse: 0.18666        valid_1's rmse: 0.214725
# [1700]  training's rmse: 0.18551        valid_1's rmse: 0.214701
# [1800]  training's rmse: 0.184437       valid_1's rmse: 0.21469
