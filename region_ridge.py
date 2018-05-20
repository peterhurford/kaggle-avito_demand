import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import rmse, normalize_text, print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runRidge5(train_X, train_y, test_X, test_y, test_X2):
    model = Ridge(alpha=5.0)
    print_step('Fit Ridge')
    model.fit(train_X, train_y)
    print_step('Ridge Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Ridge Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2

def runRidge8(train_X, train_y, test_X, test_y, test_X2):
    model = Ridge(alpha=8.0)
    print_step('Fit Ridge')
    model.fit(train_X, train_y)
    print_step('Ridge Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Ridge Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()

def run_ridge_on_region(region):
    if not is_in_cache('region_ridges_' + region):
        print_step(region + ' > Subsetting')
        train_c = train[train['region'] == region].copy()
        test_c = test[test['region'] == region].copy()
        print(train_c.shape)
        print(test_c.shape)
        target = train_c['deal_probability'].values
        train_id = train_c['item_id']
        test_id = test_c['item_id']
        train_c.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
        test_c.drop(['item_id'], axis=1, inplace=True)

        print_step(region + ' > Titlecat TFIDF 1/3')
        train_c['titlecat'] = train_c['parent_category_name'].fillna('') + ' ' + train_c['category_name'].fillna('') + ' ' + train_c['category_name'].fillna('') + ' ' + train_c['param_1'].fillna('') + ' ' + train_c['param_2'].fillna('') + ' ' + train_c['param_3'].fillna('') + ' ' + train_c['title'].fillna('')
        test_c['titlecat'] = test_c['parent_category_name'].fillna('') + ' ' + test_c['category_name'].fillna('') + ' ' + test_c['category_name'].fillna('') + ' ' + test_c['param_1'].fillna('') + ' ' + test_c['param_2'].fillna('') + ' ' + test_c['param_3'].fillna('') + ' ' + test_c['title'].fillna('')
        print_step(region + ' > Titlecat TFIDF 2/3')
        tfidf = TfidfVectorizer(ngram_range=(1, 2),
                                max_features=100000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                encoding='KOI8-R')
        tfidf_train = tfidf.fit_transform(train_c['titlecat'])
        print(tfidf_train.shape)
        print_step(region + ' > Titlecat TFIDF 3/3')
        tfidf_test = tfidf.transform(test_c['titlecat'])
        print(tfidf_test.shape)

        print_step(region + ' > Titlecat TFIDF Ridge')
        results = run_cv_model(tfidf_train, tfidf_test, target, runRidge5, rmse, region + '-titlecat-ridge')
        train_c['region_title_ridge'] = results['train']
        test_c['region_title_ridge'] = results['test']

        print_step(region + ' > Description TFIDF 1/3')
        train_c['desc'] = train_c['title'].fillna('') + ' ' + train_c['description'].fillna('')
        test_c['desc'] = test_c['title'].fillna('') + ' ' + test_c['description'].fillna('')
        print_step(region + ' > Description TFIDF 2/3')
        tfidf = TfidfVectorizer(ngram_range=(1, 2),
                                max_features=100000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                encoding='KOI8-R')
        tfidf_train2 = tfidf.fit_transform(train_c['desc'].fillna(''))
        print(tfidf_train2.shape)
        print_step(region + ' > Description TFIDF 3/3')
        tfidf_test2 = tfidf.transform(test_c['desc'].fillna(''))
        print(tfidf_test2.shape)
        results = run_cv_model(tfidf_train2, tfidf_test2, target, runRidge5, rmse, region + '-desc-ridge')
        train_c['region_desc_ridge'] = results['train']
        test_c['region_desc_ridge'] = results['test']

        print_step(region + ' > Text Char TFIDF 1/2')
        # Using char n-grams ends up being surprisingly good, HT https://www.kaggle.com/c/avito-demand-prediction/discussion/56061#325063
        tfidf = TfidfVectorizer(ngram_range=(2, 5),
                                max_features=100000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                analyzer='char',
                                encoding='KOI8-R')
        tfidf_train3 = tfidf.fit_transform(train_c['desc'])
        print(tfidf_train3.shape)
        print_step(region + ' > Text Char TFIDF 2/2')
        tfidf_test3 = tfidf.transform(test_c['desc'])
        print(tfidf_test3.shape)

        results = run_cv_model(tfidf_train3, tfidf_test3, target, runRidge5, rmse, region + '-desc-char-ridge')
        train_c['region_desc_char_ridge'] = results['train']
        test_c['region_desc_char_ridge'] = results['test']

        print_step('Merging 1/2')
        train_c2 = hstack((tfidf_train, tfidf_train2, tfidf_train3)).tocsr()
        print_step('Merging 2/2')
        test_c2 = hstack((tfidf_test, tfidf_test2, tfidf_test3)).tocsr()
        print(train_c2.shape)
        print(test_c2.shape)

        print('~~~~~~~~~~~~~~~~~~~~~~~~')
        print_step('Run Full Text Ridge')
        results = run_cv_model(train_c2, test_c2, target, runRidge8, rmse, region + '-text-ridge')
        train_c['region_all_text_ridge'] = results['train']
        test_c['region_all_text_ridge'] = results['test']

        print('~~~~~~~~~~~~~~~~~~~~~~')
        print_step(region + ' > Dropping')
        train_c.drop([c for c in train_c.columns if 'ridge' not in c], axis=1, inplace=True)
        test_c.drop([c for c in test_c.columns if 'ridge' not in c], axis=1, inplace=True)
        train_c['item_id'] = train_id
        test_c['item_id'] = test_id

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print_step(region + ' > Saving in Cache')
        save_in_cache('region_ridges_' + region, train_c, test_c)
    else:
        print(region + ' already in cache! Skipping...')
    return True

print_step('Compiling set')
regions = train['region'].unique()
n_cpu = mp.cpu_count()
n_nodes = max(n_cpu - 3, 2)
print_step('Starting a jobs server with %d nodes' % n_nodes)
pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
res = pool.map(run_ridge_on_region, regions)
pool.close()
pool.join()
pool.terminate()
pool.restart()

print('~~~~~~~~~~~~~~~~')
print_step('Merging 1/5')
pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
dfs = pool.map(lambda c: load_cache('region_ridges_' + c), regions)
pool.close()
pool.join()
pool.terminate()
pool.restart()
print_step('Merging 2/5')
train_dfs = map(lambda x: x[0], dfs)
test_dfs = map(lambda x: x[1], dfs)
print_step('Merging 3/5')
train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)
print_step('Merging 4/5')
train_ridge = train.merge(train_df, on='item_id')
print_step('Merging 5/5')
test_ridge = test.merge(test_df, on='item_id')

print_step('RMSEs')
print(rmse(train_ridge['deal_probability'], train_ridge['region_title_ridge']))     #0.2299763066787577
print(rmse(train_ridge['deal_probability'], train_ridge['region_desc_ridge']))      #0.2285397891646121
print(rmse(train_ridge['deal_probability'], train_ridge['region_desc_char_ridge'])) #0.22842621237885027
print(rmse(train_ridge['deal_probability'], train_ridge['region_all_text_ridge']))  #0.2267402863402588
import pdb
pdb.set_trace()

print('~~~~~~~~~~~~~~~')
print_step('Caching...')
save_in_cache('region_ridges', train_ridge, test_ridge)
