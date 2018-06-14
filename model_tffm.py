import tensorflow as tf

import pandas as pd

from tffm import TFFMRegressor

from utils import print_step, rmse
from cv import run_cv_model
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runTFFM3(train_X, train_y, test_X, test_y, test_X2):
    iters = 1
    rounds = 1
    model = TFFMRegressor(order=3, rank=40, 
                          optimizer=tf.train.AdamOptimizer(learning_rate=0.00005), 
                          n_epochs=rounds, batch_size=1024,
                          init_std=0.01, reg=0.1,
                          input_type='sparse', seed=52, verbose=2)
    print_step('Fit TFFM3')
    for i in range(rounds):
        model.fit(train_X, train_y.values, n_epochs=iters)
        pred_test_y = model.predict(test_X)
        print_step('Iteration {}/{} -- RMSE: {}'.format(i + 1, rounds, rmse(pred_test_y, test_y)))
    print_step('TFFM3 Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2

def runTFFM2(train_X, train_y, test_X, test_y, test_X2):
    iters = 1
    rounds = 2
    model = TFFMRegressor(order=2, rank=50, 
                          optimizer=tf.train.AdamOptimizer(learning_rate=0.00005), 
                          n_epochs=rounds, batch_size=1024,
                          init_std=0.01, reg=0.1,
                          input_type='sparse', seed=42, verbose=2)
    print_step('Fit TFFM2')
    for i in range(rounds):
        model.fit(train_X, train_y.values, n_epochs=iters)
        pred_test_y = model.predict(test_X)
        print_step('Iteration {}/{} -- RMSE: {}'.format(i + 1, rounds, rmse(pred_test_y, test_y)))
    print_step('TFFM2 Predict 2/2')
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

print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/2')
train, test = load_cache('complete_fm_data')

print('~~~~~~~~~~~~~~')
print_step('Run TFFM2')
results = run_cv_model(train, test, target, runTFFM2, rmse, 'tffm2')

print_step('Cache')
save_in_cache('tffm2', pd.DataFrame({'tffm2': results['train']}),
                       pd.DataFrame({'tffm2': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_tffm2.csv', index=False)
print_step('Done!')
# [2018-06-06 05:26:11.306486] tffm2 cv scores : [0.22541019023816908, 0.22463933727489538, 0.22452885067937228, 0.2245032642720666, 0.22523463698732962]
# [2018-06-06 05:26:11.306561] tffm2 mean cv score : 0.22486325589036663
# [2018-06-06 05:26:11.306664] tffm2 std cv score : 0.0003817385117403105


print('~~~~~~~~~~~~~~')
print_step('Run TFFM3')
results = run_cv_model(train, test, target, runTFFM3, rmse, 'tffm3')

print_step('Cache')
save_in_cache('tffm3', pd.DataFrame({'tffm3': results['train']}),
                       pd.DataFrame({'tffm3': results['test']}))
# [2018-06-06 07:08:39.924618] tffm3 cv scores : [0.22570839329650627, 0.22487660932369866, 0.22496132829464682, 0.2249749122982098, 0.2258635273882785]
# [2018-06-06 07:08:39.924683] tffm3 mean cv score : 0.225276954120268
# [2018-06-06 07:08:39.924782] tffm3 std cv score : 0.00041984112274743164
import pdb
pdb.set_trace()
