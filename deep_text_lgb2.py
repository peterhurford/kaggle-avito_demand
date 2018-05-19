import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache


# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.1,
              'application': 'regression',
			  'max_depth': 9,
              'num_leaves': 2 ** 9,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.2,
              'nthread': 3,
              'lambda_l1': 1,
              'lambda_l2': 1,
              'min_data_in_leaf': 40}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=1000,
                      valid_sets=watchlist,
                      verbose_eval=10)
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/7')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

if not is_in_cache('50k_best'):
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data 2/7')
    train_fe, test_fe = load_cache('ohe_data')

    print_step('Importing Data 3/7')
    tfidf_train, tfidf_test = load_cache('titlecat_tfidf')

    print_step('Importing Data 4/7')
    tfidf_train2, tfidf_test2 = load_cache('text_tfidf')

    print_step('Importing Data 5/7')
    tfidf_train3, tfidf_test3 = load_cache('text_char_tfidf')

    print_step('Importing Data 6/7')
    train = hstack((train_fe, tfidf_train, tfidf_train2, tfidf_train3)).tocsr()
    print_step('Importing Data 7/7')
    test = hstack((test_fe, tfidf_test, tfidf_test2, tfidf_test3)).tocsr()
    print(train.shape)
    print(test.shape)

    print_step('SelectKBest 1/2')
    fselect = SelectKBest(f_regression, k=50000)
    train = fselect.fit_transform(train, target)
    print_step('SelectKBest 2/2')
    test = fselect.transform(test)
    print(train.shape)
    print(test.shape)

    print_step('Saving to cache...')
    save_in_cache('50k_best', train, test)
else:
    print_step('Loading from cache...')
    train, test = load_cache('50k_best')


print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train, test, target, runLGB, rmse, 'lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('deep_text_lgb', pd.DataFrame({'deep_text_lgb': results['train']}),
							   pd.DataFrame({'deep_text_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb7.csv', index=False)
print_step('Done!')

# Deep LGB                                                       - Dim 195673, 5CV 0.22196

# CURRENT
# [2018-05-03 09:16:12.670379] lgb cv scores : [0.22257689498410263, 0.22140350676341775, 0.22190683421601698, 0.22168666577421173, 0.22221717818648018]
# [2018-05-03 09:16:12.671351] lgb mean cv score : 0.22195821598484583
# [2018-05-03 09:16:12.672767] lgb std cv score : 0.00040838879744612577

# [10]    training's rmse: 0.240701       valid_1's rmse: 0.241443
# [100]   training's rmse: 0.224196       valid_1's rmse: 0.227253
# [200]   training's rmse: 0.221323       valid_1's rmse: 0.225548
# [300]   training's rmse: 0.21958        valid_1's rmse: 0.224654
# [400]   training's rmse: 0.218335       valid_1's rmse: 0.224104
# [500]   training's rmse: 0.217273       valid_1's rmse: 0.223712
# [600]   training's rmse: 0.216378       valid_1's rmse: 0.223389
# [700]   training's rmse: 0.215555       valid_1's rmse: 0.223133
# [800]   training's rmse: 0.214814       valid_1's rmse: 0.222927
# [900]   training's rmse: 0.21409        valid_1's rmse: 0.222717
# [1000]  training's rmse: 0.213506       valid_1's rmse: 0.222577
