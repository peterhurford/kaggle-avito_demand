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
              'nthread': 16,
              'lambda_l1': 1,
              'lambda_l2': 1,
              'min_data_in_leaf': 40}
	import pdb
	pdb.set_trace()
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
print_step('Importing Data 1/2')
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
    print_step('Importing Data 2/2')
    train, test = load_cache('complete_ridge_data')
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
results = run_cv_model(train, test, target, runLGB, rmse, 'deep_text_lgb2')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('deep_text_lgb2', pd.DataFrame({'deep_text_lgb2': results['train']}),
							    pd.DataFrame({'deep_text_lgb2': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_deep_text_lgb2.csv', index=False)
print_step('Done!')

# CURRENT
# [2018-06-06 21:16:40.327713] deep_text_lgb2 cv scores : [0.2218181252183542, 0.22115275787193545, 0.2211522340086538, 0.22123956148843887, 0.22168721776583822]
# [2018-06-06 21:16:40.327784] deep_text_lgb2 mean cv score : 0.2214099792706441
# [2018-06-06 21:16:40.327883] deep_text_lgb2 std cv score : 0.0002846337950326723
