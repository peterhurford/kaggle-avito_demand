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
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 4/12 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 4/12 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 5/12 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 5/12 2/3')
train_fe['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 5/12 3/3')
test_fe['complete_ridge'] = test_complete_ridge['complete_ridge']

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
print_step('Importing Data 12/12 2/4')
cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
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

print_step('Importing Data 12/12 1/5')
train_active_feats, test_active_feats = load_cache('active_feats')
train_active_feats.fillna(0, inplace=True)
test_active_feats.fillna(0, inplace=True)
print_step('Importing Data 12/12 2/5')
train_fe = pd.concat([train_fe, train_active_feats], axis=1)
print_step('Importing Data 12/12 3/5')
test_fe = pd.concat([test_fe, test_active_feats], axis=1)
print_step('Importing Data 12/12 4/5')
train_fe['user_items_per_day'] = train_fe['n_user_items'] / train_fe['user_num_days']
test_fe['user_items_per_day'] = test_fe['n_user_items'] / test_fe['user_num_days']
print_step('Importing Data 12/12 5/5')
train_fe.drop('user_id', axis=1, inplace=True)
test_fe.drop('user_id', axis=1, inplace=True)


print_step('Importing Data 16/15 1/4 NIMA Features')
train, test = get_data()
train_nima, test_nima = load_cache('img_nima')
train = train.merge(train_nima, on = 'image', how = 'left')
test = test.merge(test_nima, on = 'image', how = 'left')
cols = ["mobile_mean", "mobile_std","inception_mean", "inception_std", "nasnet_mean", "nasnet_std"]
train_fe[cols] = train[cols].fillna(0)
test_fe[cols] = test[cols].fillna(0)
del train, test, train_nima, test_nima



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


print('~~~~~~~~~~~~~~~~~~')
print_step('Run Ridge LGB')
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


print('~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 4/12 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 4/12 2/3')
train_fe['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 4/12 3/3')
test_fe['deep_lgb'] = test_deep_lgb['deep_lgb']

print_step('Run Stack LGB')
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


# Changed By Sijun
# + NIMA features
# Public LB 0.2187
# [2018-06-11 00:29:35.356944] lgb cv score 5 : 0.21435439465470424
# [2018-06-11 00:29:35.357184] lgb cv scores : [0.214839234040873, 0.21397466737613413, 0.21390426736833554, 0.21385081987088378, 0.21435439465470424]
# [2018-06-11 00:29:35.357588] lgb mean cv score : 0.21418467666218613
# [2018-06-11 00:29:35.358510] lgb std cv score : 0.0003718718027445342


# CURRENT
# [2018-06-06 17:32:25.455448] lgb cv scores : [0.  21492464321752142, 0.21407966462295236, 0.21400264246446363, 0.21395136575143095, 0.2146172980164727]
# [2018-06-06 17:32:25.455516] lgb mean cv score : 0.2143151228145682
# [2018-06-06 17:32:25.455621] lgb std cv score : 0.00038684071772371374

# [100]   training's rmse: 0.21685        valid_1's rmse: 0.220547
# [200]   training's rmse: 0.212067       valid_1's rmse: 0.218429
# [300]   training's rmse: 0.208733       valid_1's rmse: 0.217312
# [400]   training's rmse: 0.206092       valid_1's rmse: 0.216786
# [500]   training's rmse: 0.2038 valid_1's rmse: 0.216327
# [600]   training's rmse: 0.20175        valid_1's rmse: 0.21601
# [700]   training's rmse: 0.199841       valid_1's rmse: 0.215774
# [800]   training's rmse: 0.197996       valid_1's rmse: 0.215604
# [900]   training's rmse: 0.196292       valid_1's rmse: 0.215468
# [1000]  training's rmse: 0.194739       valid_1's rmse: 0.215356
# [1100]  training's rmse: 0.1933 valid_1's rmse: 0.215254
# [1200]  training's rmse: 0.191843       valid_1's rmse: 0.215168
# [1300]  training's rmse: 0.190445       valid_1's rmse: 0.215092
# [1400]  training's rmse: 0.189061       valid_1's rmse: 0.215056
# [1500]  training's rmse: 0.187763       valid_1's rmse: 0.215027
# [1600]  training's rmse: 0.186473       valid_1's rmse: 0.214986
# [1700]  training's rmse: 0.185311       valid_1's rmse: 0.214943
# [1800]  training's rmse: 0.18417        valid_1's rmse: 0.214925


# WITH LGB
# [2018-06-12 18:14:20.895038] lgb cv scores : [0.21443258908336837, 0.21361589672535733, 0.21351838218196928, 0.21366236944769273, 0.21415793413155496]
# [2018-06-12 18:14:20.895102] lgb mean cv score : 0.2138774343139885
# [2018-06-12 18:14:20.895202] lgb std cv score : 0.00035509159553822364

# [100]   training's rmse: 0.215619       valid_1's rmse: 0.219281
# [200]   training's rmse: 0.210747       valid_1's rmse: 0.216965
# [300]   training's rmse: 0.207761       valid_1's rmse: 0.216081
# [400]   training's rmse: 0.2051 valid_1's rmse: 0.21549
# [500]   training's rmse: 0.202729       valid_1's rmse: 0.21508
# [600]   training's rmse: 0.200674       valid_1's rmse: 0.214843
# [700]   training's rmse: 0.198865       valid_1's rmse: 0.214733
# [800]   training's rmse: 0.197251       valid_1's rmse: 0.214651
# [900]   training's rmse: 0.195678       valid_1's rmse: 0.214566
# [1000]  training's rmse: 0.194095       valid_1's rmse: 0.214522
# [1100]  training's rmse: 0.192649       valid_1's rmse: 0.214487
# [1200]  training's rmse: 0.19139        valid_1's rmse: 0.214475
# [1300]  training's rmse: 0.190123       valid_1's rmse: 0.214465
# [1400]  training's rmse: 0.188896       valid_1's rmse: 0.214455
# [1500]  training's rmse: 0.187758       valid_1's rmse: 0.21444
# [1600]  training's rmse: 0.186662       valid_1's rmse: 0.21443
# [1700]  training's rmse: 0.185524       valid_1's rmse: 0.214435
# [1800]  training's rmse: 0.184465       valid_1's rmse: 0.214433
