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
              'feature_fraction': 0.1,
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

print_step('Importing Data 3/15 1/4')
train_img, test_img = load_cache('img_data')
print_step('Importing Data 3/15 2/4')
cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
train_img = train_img[cols].fillna(0)
test_img = test_img[cols].fillna(0)
print_step('Importing Data 3/15 3/4')
train_fe = pd.concat([train_fe, train_img], axis=1)
print_step('Importing Data 3/15 4/4')
test_fe = pd.concat([test_fe, test_img], axis=1)

print_step('Importing Data 4/15 1/4')
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
print_step('Importing Data 4/15 2/4')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 4/15 3/4')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 4/15 4/4')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

print_step('Importing Data 5/15 1/3')
region_macro = pd.read_csv('region_macro.csv')
print_step('Importing Data 5/15 2/3')
train_fe = train_fe.merge(region_macro, how='left', on='region')
print_step('Importing Data 5/15 3/3')
test_fe = test_fe.merge(region_macro, how='left', on='region')


print_step('Importing Data 6/15 1/5')
train_active_feats, test_active_feats = load_cache('active_feats')
train_active_feats.fillna(0, inplace=True)
test_active_feats.fillna(0, inplace=True)
print_step('Importing Data 6/15 2/5')
train_fe = pd.concat([train_fe, train_active_feats], axis=1)
print_step('Importing Data 6/15 3/5')
test_fe = pd.concat([test_fe, test_active_feats], axis=1)
print_step('Importing Data 6/15 4/5')
train_fe['user_items_per_day'] = train_fe['n_user_items'] / train_fe['user_num_days']
test_fe['user_items_per_day'] = test_fe['n_user_items'] / test_fe['user_num_days']
train_fe['img_size_ratio'] = train_fe['img_file_size'] / (train_fe['img_size_x'] * train_fe['img_size_y'])
test_fe['img_size_ratio'] = test_fe['img_file_size'] / (test_fe['img_size_x'] * test_fe['img_size_y'])
print_step('Importing Data 6/15 5/5')
train_fe.drop('user_id', axis=1, inplace=True)
test_fe.drop('user_id', axis=1, inplace=True)


print_step('Importing Data 7/15 1/8')
train, test = get_data()
print_step('Importing Data 7/15 2/8')
train_nima, test_nima = load_cache('img_nima')
print_step('Importing Data 7/15 3/8')
train = train.merge(train_nima, on = 'image', how = 'left')
print_step('Importing Data 7/15 4/8')
test = test.merge(test_nima, on = 'image', how = 'left')
print_step('Importing Data 7/15 5/8')
cols = ['mobile_mean', 'mobile_std', 'inception_mean', 'inception_std',
		'nasnet_mean', 'nasnet_std']
train_fe[cols] = train[cols].fillna(0)
test_fe[cols] = test[cols].fillna(0)
print_step('Importing Data 7/15 6/8')
train_nima, test_nima = load_cache('img_nima_softmax')
print_step('Importing Data 7/15 7/8')
train = train.merge(train_nima, on = 'image', how = 'left')
test = test.merge(test_nima, on = 'image', how = 'left')
print_step('Importing Data 7/15 8/8')
cols = [x + '_' + str(y) for x in ['mobile', 'inception', 'nasnet'] for y in range(1, 11)]
train_fe[cols] = train[cols].fillna(0)
test_fe[cols] = test[cols].fillna(0)
del train, test, train_nima, test_nima


print_step('Importing Data 8/15 1/2')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 8/15 2/2')
cols = [c for c in train_ridge.columns if 'svd' in c or 'tfidf' in c]
train_fe[cols] = train_ridge[cols]
test_fe[cols] = test_ridge[cols]


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
print(train_fe.shape)
print(test_fe.shape)
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'base_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('base_lgb', pd.DataFrame({'base_lgb': results['train']}),
                          pd.DataFrame({'base_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_base_lgb.csv', index=False)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 8/15')
drops = [x + '_' + str(y) for x in ['mobile', 'inception', 'nasnet'] for y in range(1, 11)] + ['img_size_ratio', 'times_put_up_min', 'times_put_up_max']
train_fe.drop(drops, axis=1, inplace=True)
test_fe.drop(drops, axis=1, inplace=True)

print_step('Importing Data 9/15 1/2')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 9/15 2/2')
cols = [c for c in train_ridge.columns if 'ridge' in c]
train_fe[cols] = train_ridge[cols]
test_fe[cols] = test_ridge[cols]

print_step('Importing Data 10/15 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 10/15 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 10/15 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 11/15 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 11/15 2/3')
train_fe['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 11/15 3/3')
test_fe['complete_ridge'] = test_complete_ridge['complete_ridge']

print_step('Importing Data 12/15 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 12/15 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 12/15 3/4')
train_fe = pd.concat([train_fe, train_pcat_ridge], axis=1)
print_step('Importing Data 12/15 4/4')
test_fe = pd.concat([test_fe, test_pcat_ridge], axis=1)

print_step('Importing Data 13/15 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 13/15 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 13/15 3/4')
train_fe = pd.concat([train_fe, train_rcat_ridge], axis=1)
print_step('Importing Data 13/15 4/4')
test_fe = pd.concat([test_fe, test_rcat_ridge], axis=1)

print_step('Importing Data 14/15 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 14/15 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 14/15 3/4')
train_fe = pd.concat([train_fe, train_catb_ridge], axis=1)
print_step('Importing Data 14/15 4/4')
test_fe = pd.concat([test_fe, test_catb_ridge], axis=1)

print('~~~~~~~~~~~~~~~~~~')
print_step('Run Ridge LGB')
print(train_fe.shape)
print(test_fe.shape)
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'ridge_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('ridge_lgb', pd.DataFrame({'ridge_lgb': results['train']}),
                           pd.DataFrame({'ridge_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_ridge_lgb.csv', index=False)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 15/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 15/15 2/3')
train_fe['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 15/15 3/3')
test_fe['deep_lgb'] = test_deep_lgb['deep_lgb']

print('~~~~~~~~~~~~~~~~~~')
print_step('Run Stack LGB')
print(train_fe.shape)
print(test_fe.shape)
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'stack_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('stack_lgb', pd.DataFrame({'stack_lgb': results['train']}),
                           pd.DataFrame({'stack_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_stack_lgb.csv', index=False)
print_step('Done!')


# BASE
# [2018-06-17 02:15:35.160780] base_lgb cv scores : [0.2180584840223271, 0.2173285470396579, 0.21730824294582546, 0.21708803739182905, 0.21764331398813264]
# [2018-06-17 02:15:35.160849] base_lgb mean cv score : 0.21748532507755441
# [2018-06-17 02:15:35.160952] base_lgb std cv score : 0.0003368223898769227


# RIDGE LGB
# [2018-06-15 04:56:50.288476] ridge_lgb cv scores : [0.2147783829876306, 0.21395614971673593, 0.2138701184179281, 0.21380657669625333, 0.21439726308695933]
# [2018-06-15 04:56:50.288546] ridge_lgb mean cv score : 0.21416169818110148
# [2018-06-15 04:56:50.288657] ridge_lgb std cv score : 0.00037126033291255727

# [100]   training's rmse: 0.216829       valid_1's rmse: 0.220427
# [200]   training's rmse: 0.212227       valid_1's rmse: 0.218217
# [300]   training's rmse: 0.208823       valid_1's rmse: 0.217209
# [400]   training's rmse: 0.2061 valid_1's rmse: 0.216543
# [500]   training's rmse: 0.203779       valid_1's rmse: 0.216114
# [600]   training's rmse: 0.201708       valid_1's rmse: 0.2158
# [700]   training's rmse: 0.199802       valid_1's rmse: 0.215607
# [800]   training's rmse: 0.197992       valid_1's rmse: 0.215438
# [900]   training's rmse: 0.19635        valid_1's rmse: 0.215269
# [1000]  training's rmse: 0.194769       valid_1's rmse: 0.215163
# [1100]  training's rmse: 0.193348       valid_1's rmse: 0.215067
# [1200]  training's rmse: 0.19194        valid_1's rmse: 0.214993
# [1300]  training's rmse: 0.190427       valid_1's rmse: 0.214965
# [1400]  training's rmse: 0.189121       valid_1's rmse: 0.214912
# [1500]  training's rmse: 0.187851       valid_1's rmse: 0.214862
# [1600]  training's rmse: 0.186619       valid_1's rmse: 0.214822
# [1700]  training's rmse: 0.185392       valid_1's rmse: 0.214793
# [1800]  training's rmse: 0.18423        valid_1's rmse: 0.214778


# WITH LGB
# [2018-06-15 06:27:37.917003] stack_lgb cv scores : [0.21435706607494084, 0.21355795065997074, 0.2134008730635741, 0.21346079057184786, 0.21400001188721424]
# [2018-06-15 06:27:37.917069] stack_lgb mean cv score : 0.21375533845150957
# [2018-06-15 06:27:37.917175] stack_lgb std cv score : 0.00036696248523406254

# [100]   training's rmse: 0.21588        valid_1's rmse: 0.219561
# [200]   training's rmse: 0.210884       valid_1's rmse: 0.216986
# [300]   training's rmse: 0.2075 valid_1's rmse: 0.21588
# [400]   training's rmse: 0.204813       valid_1's rmse: 0.215324
# [500]   training's rmse: 0.202693       valid_1's rmse: 0.215015
# [600]   training's rmse: 0.200695       valid_1's rmse: 0.214843
# [700]   training's rmse: 0.198797       valid_1's rmse: 0.214715
# [800]   training's rmse: 0.197068       valid_1's rmse: 0.214627
# [900]   training's rmse: 0.195469       valid_1's rmse: 0.214562
# [1000]  training's rmse: 0.19397        valid_1's rmse: 0.2145
# [1100]  training's rmse: 0.192422       valid_1's rmse: 0.214478
# [1200]  training's rmse: 0.191044       valid_1's rmse: 0.214446
# [1300]  training's rmse: 0.18968        valid_1's rmse: 0.214414
# [1400]  training's rmse: 0.188452       valid_1's rmse: 0.21439
# [1500]  training's rmse: 0.187219       valid_1's rmse: 0.214378
# [1600]  training's rmse: 0.186012       valid_1's rmse: 0.214378
# [1700]  training's rmse: 0.184909       valid_1's rmse: 0.214364
# [1800]  training's rmse: 0.18381        valid_1's rmse: 0.214357
