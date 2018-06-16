import gc

import pandas as pd
import numpy as np

from scipy.sparse import hstack

import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

from cv import run_cv_model
from utils import rmse, print_step, normalize_text, bin_and_ohe_data
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runFM(train_X, train_y, test_X, test_y, test_X2):
    iters = 1
    rounds = 4 
    model = FM_FTRL(alpha=0.005, beta=0.01, L1=0.0001, L2=0.1,
                    alpha_fm=0.005, L2_fm=0.0, init_fm=0.01, D_fm=50, e_noise=0.0, 
                    D=train_X.shape[1], iters=iters, inv_link='identity', threads=16,
                    use_avx=1, verbose=0)
    print_step('Fit FM')
    for i in range(rounds):
        model.fit(train_X, train_y, reset=False)
        pred_test_y = model.predict(test_X)
        print_step('Iteration {}/{} -- RMSE: {}'.format(i + 1, rounds, rmse(pred_test_y, test_y)))
    print_step('FM Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/10')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

if not is_in_cache('titlecat_wordbatch') or not is_in_cache('text_wordbatch'):
    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Titlecat Wordbatch 1/5')
    train['titlecat'] = train['parent_category_name'].fillna('') + ' ' + train['category_name'].fillna('') + ' ' + train['param_1'].fillna('') + ' ' + train['param_2'].fillna('') + ' ' + train['param_3'].fillna('') + ' ' + train['title'].fillna('')
    test['titlecat'] = test['parent_category_name'].fillna('') + ' ' + test['category_name'].fillna('') + ' ' + test['param_1'].fillna('') + ' ' + test['param_2'].fillna('') + ' ' + test['param_3'].fillna('') + ' ' + test['title'].fillna('')
    if not is_in_cache('titlecat_wordbatch'):
        print_step('Titlecat Wordbatch 2/5')
        wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                      "hash_ngrams_weights": [1.5, 1.0],
                                                                      "hash_size": 2 ** 29,
                                                                      "norm": None,
                                                                      "tf": 'binary',
                                                                      "idf": None,
                                                                      }), procs=8)
        wb.dictionary_freeze = True
        wordbatch_train = wb.fit_transform(train['titlecat'])
        print(wordbatch_train.shape)
        print_step('Titlecat Wordbatch 3/5')
        wordbatch_test = wb.transform(test['titlecat'])
        print(wordbatch_test.shape)
        del(wb)
        gc.collect()
        print_step('Titlecat Wordbatch 4/5')
        mask = np.where(wordbatch_train.getnnz(axis=0) > 3)[0]
        wordbatch_train = wordbatch_train[:, mask]
        print(wordbatch_train.shape)
        print_step('Titlecat Wordbatch 5/5')
        wordbatch_test = wordbatch_test[:, mask]
        print(wordbatch_test.shape)
        print_step('Saving to cache...')
        save_in_cache('titlecat_wordbatch', wordbatch_train, wordbatch_test)
    else:
        print_step('Loading from cache...')
        wordbatch_train, wordbatch_test = load_cache('titlecat_wordbatch')

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Text Wordbatch 1/5')
    train['desc'] = train['title'].fillna('') + ' ' + train['description'].fillna('')
    test['desc'] = test['title'].fillna('') + ' ' + test['description'].fillna('')
    if not is_in_cache('text_wordbatch'):
        print_step('Text Wordbatch 2/5')
        wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                      "hash_ngrams_weights": [1.0, 1.0],
                                                                      "hash_size": 2 ** 28,
                                                                      "norm": "l2",
                                                                      "tf": 1.0,
                                                                      "idf": None}), procs=8)
        wb.dictionary_freeze = True
        wordbatch_train = wb.fit_transform(train['desc'].fillna(''))
        print(wordbatch_train.shape)
        print_step('Text Wordbatch 3/5')
        wordbatch_test = wb.transform(test['desc'].fillna(''))
        print(wordbatch_test.shape)
        del(wb)
        gc.collect()
        print_step('Text Wordbatch 4/5')
        mask = np.where(wordbatch_train.getnnz(axis=0) > 8)[0]
        wordbatch_train = wordbatch_train[:, mask]
        print(wordbatch_train.shape)
        print_step('Text Wordbatch 5/5')
        wordbatch_test = wordbatch_test[:, mask]
        print(wordbatch_test.shape)
        print_step('Saving to cache...')
        save_in_cache('text_wordbatch', wordbatch_train, wordbatch_test)
    else:
        print_step('Loading from cache...')
        wordbatch_train, wordbatch_test = load_cache('text_wordbatch')

if not is_in_cache('complete_fm'):
    if not is_in_cache('complete_fm_data'):
        print('~~~~~~~~~~~~~~~~~~~~~~~')
        print_step('Importing Data 2/10')
        train_fe, test_fe = load_cache('ohe_data')

        print_step('Importing Data 3/10')
        wordbatch_train, wordbatch_test = load_cache('titlecat_wordbatch')

        print_step('Importing Data 4/10')
        wordbatch_train2, wordbatch_test2 = load_cache('text_wordbatch')

        print_step('Importing Data 5/10')
        train = hstack((wordbatch_train, wordbatch_train2)).tocsr()
        print_step('Importing Data 6/10')
        test = hstack((wordbatch_test, wordbatch_test2)).tocsr()
        print(train.shape)
        print(test.shape)

        print_step('Importing Data 7/10 1/5')
        train_img, test_img = load_cache('img_data')
        print_step('Importing Data 7/10 2/5')
        drops = ['item_id', 'img_path', 'img_std_color', 'img_sum_color', 'img_rms_color',
                 'img_var_color', 'img_average_color', 'deal_probability']
        drops += [c for c in train_img if 'hist' in c]
        dummy_cols = ['img_average_color']
        numeric_cols = list(set(train_img.columns) - set(drops) - set(dummy_cols))
        train_img = train_img[numeric_cols + dummy_cols].fillna(0)
        test_img = test_img[numeric_cols + dummy_cols].fillna(0)
        print_step('Importing Data 7/10 3/5')
        train_img, test_img = bin_and_ohe_data(train_img, test_img,
                                               numeric_cols=numeric_cols,
                                               dummy_cols=dummy_cols)
        print_step('Importing Data 7/10 4/5')
        train = hstack((train, train_img)).tocsr()
        print(train.shape)
        print_step('Importing Data 7/10 5/5')
        test = hstack((test, test_img)).tocsr()
        print(test.shape)

        print_step('GC')
        del test_img
        del train_img
        del wordbatch_test
        del wordbatch_test2
        del wordbatch_train
        del wordbatch_train2
        gc.collect()

        print_step('Importing Data 8/10 1/8')
        train_, test_ = get_data()
        train_['city'] = train_['city'] + ', ' + train_['region']
        test_['city'] = test_['city'] + ', ' + test_['region']
        print_step('Importing Data 8/10 2/8')
        # HT: https://www.kaggle.com/jpmiller/russian-cities/data
        # HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
        locations = pd.read_csv('city_latlons.csv')
        print_step('Importing Data 8/10 3/8')
        train_ = train_.merge(locations, how='left', left_on='city', right_on='location')
        print_step('Importing Data 8/10 4/8')
        test_ = test_.merge(locations, how='left', left_on='city', right_on='location')
        print_step('Importing Data 8/10 5/8')
        train_ = train_[['lon', 'lat']]
        test_ = test_[['lon', 'lat']]
        print_step('Importing Data 8/10 6/8')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=['lon', 'lat'],
                                         dummy_cols=[])
        print_step('Importing Data 8/10 7/8')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 8/10 8/8')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Importing Data 9/10 1/8')
        train_, test_ = get_data()
        print_step('Importing Data 9/10 2/8')
        region_macro = pd.read_csv('region_macro.csv')
        print_step('Importing Data 9/10 3/8')
        train_ = train_.merge(region_macro, how='left', on='region')
        print_step('Importing Data 9/10 4/8')
        test_ = test_.merge(region_macro, how='left', on='region')
        print_step('Importing Data 9/10 5/8')
        cols = ['unemployment_rate', 'GDP_PC_PPP', 'HDI']
        train_ = train_[cols]
        test_ = test_[cols]
        print_step('Importing Data 9/10 6/8')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=cols,
                                         dummy_cols=[])
        print_step('Importing Data 9/10 7/8')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 9/10 8/8')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Importing Data 10/10 1/4')
        train_, test_ = load_cache('active_feats')
        train_.fillna(0, inplace=True)
        test_.fillna(0, inplace=True)
        train_.drop('user_id', axis=1, inplace=True)
        test_.drop('user_id', axis=1, inplace=True)
        print_step('Importing Data 10/10 2/4')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=train_.columns.values.tolist(),
                                         dummy_cols=[])
        print_step('Importing Data 10/10 3/4')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 10/10 4/4')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Caching')
        save_in_cache('complete_fm_data', train, test)
    else:
        train, test = load_cache('complete_fm_data')

    print_step('Run Complete FM')
    results = run_cv_model(train, test, target, runFM, rmse, 'complete-fm')
    import pdb
    pdb.set_trace()

    print('~~~~~~~~~~')
    print_step('Cache')
    save_in_cache('complete_fm', pd.DataFrame({'complete_fm': results['train']}),
                                 pd.DataFrame({'complete_fm': results['test']}))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Prepping submission file')
    submission = pd.DataFrame()
    submission['item_id'] = test_id
    submission['deal_probability'] = results['test'].clip(0.0, 1.0)
    submission.to_csv('submit/submit_fm.csv', index=False)
    print_step('Done!')
# [2018-06-06 00:32:04.610346] complete-fm cv scores : [0.22574842632640188, 0.22496819763704198, 0.22500458473036733, 0.22493600574130557, 0.22553416531004894]
# [2018-06-06 00:32:04.610419] complete-fm mean cv score : 0.22523827594903315
# [2018-06-06 00:32:04.610518] complete-fm std cv score : 0.00033666751558799967
