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


print_step('Importing Data 7/15')
train, test = get_data()
train_nima, test_nima = load_cache('img_nima')
train = train.merge(train_nima, on = 'image', how = 'left')
test = test.merge(test_nima, on = 'image', how = 'left')
cols = ['mobile_mean', 'mobile_std', 'inception_mean', 'inception_std',
		'nasnet_mean', 'nasnet_std']
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
drops = ['img_size_ratio', 'times_put_up_min', 'times_put_up_max']
train_fe.drop(drops, axis=1, inplace=True)
test_fe.drop(drops, axis=1, inplace=True)

print_step('Importing Data 9/15 1/3')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 9/15 2/3')
train_fe = pd.concat([train_fe, train_ridge], axis=1)
print_step('Importing Data 9/15 3/3')
test_fe = pd.concat([test_fe, test_ridge], axis=1)

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
print_step('Importing Data 15/15 1/3')
train_deep_lgb, test_deep_lgb = load_cache('deep_lgb')
print_step('Importing Data 15/15 2/3')
train_fe['deep_lgb'] = train_deep_lgb['deep_lgb']
print_step('Importing Data 15/15 3/3')
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


# BASE
# [2018-06-14 15:36:12.468047] base_lgb cv scores : [0.21970573671775664, 0.2189724471126797, 0.2189077547425295, 0.21882077122829713, 0.2193020339508764]
# [2018-06-14 15:36:12.468113] base_lgb mean cv score : 0.21914174875042786
# [2018-06-14 15:36:12.468214] base_lgb std cv score : 0.0003256429279945189

# [100]   training's rmse: 0.226989       valid_1's rmse: 0.229376
# [200]   training's rmse: 0.221307       valid_1's rmse: 0.225421
# [300]   training's rmse: 0.218186       valid_1's rmse: 0.223813
# [400]   training's rmse: 0.215975       valid_1's rmse: 0.222886
# [500]   training's rmse: 0.213945       valid_1's rmse: 0.222142
# [600]   training's rmse: 0.212382       valid_1's rmse: 0.221699
# [700]   training's rmse: 0.210814       valid_1's rmse: 0.22135
# [800]   training's rmse: 0.20937        valid_1's rmse: 0.221082
# [900]   training's rmse: 0.207895       valid_1's rmse: 0.220803
# [1000]  training's rmse: 0.206642       valid_1's rmse: 0.220603
# [1100]  training's rmse: 0.205368       valid_1's rmse: 0.220419
# [1200]  training's rmse: 0.204154       valid_1's rmse: 0.220254
# [1300]  training's rmse: 0.202948       valid_1's rmse: 0.220114
# [1400]  training's rmse: 0.201868       valid_1's rmse: 0.219987
# [1500]  training's rmse: 0.200826       valid_1's rmse: 0.219903
# [1600]  training's rmse: 0.199883       valid_1's rmse: 0.219826
# [1700]  training's rmse: 0.198968       valid_1's rmse: 0.219772
# [1800]  training's rmse: 0.197979       valid_1's rmse: 0.219706


# RIDGE LGB
# [2018-06-13 21:06:33.307233] ridge_lgb cv scores : [0.21481023099930235, 0.21397466737613413, 0.21390426736833568, 0.21385081987088378, 0.21435439465470424]
# [2018-06-13 21:06:33.307298] ridge_lgb mean cv score : 0.214178876053872
# [2018-06-13 21:06:33.307399] ridge_lgb std cv score : 0.0003617036963564816

# [100]   training's rmse: 0.216785       valid_1's rmse: 0.220499
# [200]   training's rmse: 0.212039       valid_1's rmse: 0.218252
# [300]   training's rmse: 0.208943       valid_1's rmse: 0.217347
# [400]   training's rmse: 0.206321       valid_1's rmse: 0.216727
# [500]   training's rmse: 0.204102       valid_1's rmse: 0.216338
# [600]   training's rmse: 0.201907       valid_1's rmse: 0.215948
# [700]   training's rmse: 0.200053       valid_1's rmse: 0.215731
# [800]   training's rmse: 0.198012       valid_1's rmse: 0.215524
# [900]   training's rmse: 0.19636        valid_1's rmse: 0.215367
# [1000]  training's rmse: 0.194721       valid_1's rmse: 0.215224
# [1100]  training's rmse: 0.193234       valid_1's rmse: 0.215152
# [1200]  training's rmse: 0.191779       valid_1's rmse: 0.215059
# [1300]  training's rmse: 0.190345       valid_1's rmse: 0.215
# [1400]  training's rmse: 0.189003       valid_1's rmse: 0.21494
# [1500]  training's rmse: 0.187796       valid_1's rmse: 0.214898
# [1600]  training's rmse: 0.186545       valid_1's rmse: 0.214885
# [1700]  training's rmse: 0.185354       valid_1's rmse: 0.214834
# [1800]  training's rmse: 0.18423        valid_1's rmse: 0.21481


# WITH LGB
# [2018-06-13 22:56:14.977763] stack_lgb cv scores : [0.21445831192298195, 0.2135156996792157, 0.21343223308771092, 0.21356783859418368, 0.21405113908072435]
# [2018-06-13 22:56:14.977830] stack_lgb mean cv score : 0.21380504447296328
# [2018-06-13 22:56:14.977935] stack_lgb std cv score : 0.00039148340572575814

# [100]   training's rmse: 0.215772       valid_1's rmse: 0.219468
# [200]   training's rmse: 0.210742       valid_1's rmse: 0.216861
# [300]   training's rmse: 0.207539       valid_1's rmse: 0.215862
# [400]   training's rmse: 0.204946       valid_1's rmse: 0.215364
# [500]   training's rmse: 0.202645       valid_1's rmse: 0.215013
# [600]   training's rmse: 0.200699       valid_1's rmse: 0.214837
# [700]   training's rmse: 0.198963       valid_1's rmse: 0.214733
# [800]   training's rmse: 0.197231       valid_1's rmse: 0.214655
# [900]   training's rmse: 0.195496       valid_1's rmse: 0.214603
# [1000]  training's rmse: 0.193909       valid_1's rmse: 0.214567
# [1100]  training's rmse: 0.192581       valid_1's rmse: 0.214539
# [1200]  training's rmse: 0.191275       valid_1's rmse: 0.214518
# [1300]  training's rmse: 0.18995        valid_1's rmse: 0.21451
# [1400]  training's rmse: 0.188685       valid_1's rmse: 0.214474
# [1500]  training's rmse: 0.187465       valid_1's rmse: 0.214458
# [1600]  training's rmse: 0.186386       valid_1's rmse: 0.214457
# [1700]  training's rmse: 0.185242       valid_1's rmse: 0.214448
# [1800]  training's rmse: 0.184096       valid_1's rmse: 0.214458
