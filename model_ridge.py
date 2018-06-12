import gc

import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import rmse, print_step, bin_and_ohe_data
from cache import get_data, is_in_cache, load_cache, save_in_cache


NCOMP = 20

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

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

if not is_in_cache('tfidf_ridges') or not is_in_cache('titlecat_tfidf') or not is_in_cache('text_tfidf') or not is_in_cache('text_char_tfidf'):
    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Title TFIDF 1/2')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train['title'])
    print(tfidf_train.shape)
    print_step('Title TFIDF 2/2')
    tfidf_test = tfidf.transform(test['title'])
    print(tfidf_test.shape)

    print_step('Title SVD 1/4')
    svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Title SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Title SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Title SVD 4/4')
    train_svd.columns = ['svd_title_'+str(i+1) for i in range(NCOMP)]
    test_svd.columns = ['svd_title_'+str(i+1) for i in range(NCOMP)]
    train = pd.concat([train, train_svd], axis=1)
    test = pd.concat([test, test_svd], axis=1)

    print_step('Titlecat TFIDF 1/3')
    train['titlecat'] = train['parent_category_name'].fillna('') + ' ' + train['category_name'].fillna('') + ' ' + train['param_1'].fillna('') + ' ' + train['param_2'].fillna('') + ' ' + train['param_3'].fillna('') + ' ' + train['title'].fillna('')
    test['titlecat'] = test['parent_category_name'].fillna('') + ' ' + test['category_name'].fillna('') + ' ' + test['param_1'].fillna('') + ' ' + test['param_2'].fillna('') + ' ' + test['param_3'].fillna('') + ' ' + test['title'].fillna('')
    if not is_in_cache('titlecat_tfidf'):
        print_step('Titlecat TFIDF 2/3')
        tfidf = TfidfVectorizer(ngram_range=(1, 2),
                                max_features=300000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                encoding='KOI8-R')
        tfidf_train = tfidf.fit_transform(train['titlecat'])
        print(tfidf_train.shape)
        print_step('Titlecat TFIDF 3/3')
        tfidf_test = tfidf.transform(test['titlecat'])
        print(tfidf_test.shape)
        print_step('Saving to cache...')
        save_in_cache('titlecat_tfidf', tfidf_train, tfidf_test)
    else:
        print_step('Loading from cache...')
        tfidf_train, tfidf_test = load_cache('titlecat_tfidf')

    print_step('Titlecat SVD 1/4')
    svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Titlecat SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Titlecat SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Titlecat SVD 4/4')
    train_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(NCOMP)]
    test_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(NCOMP)]
    train = pd.concat([train, train_svd], axis=1)
    test = pd.concat([test, test_svd], axis=1)

    print_step('Titlecat TFIDF Ridge')
    results = run_cv_model(tfidf_train, tfidf_test, target, runRidge5, rmse, 'titlecat-ridge')
    train['title_ridge'] = results['train']
    test['title_ridge'] = results['test']

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Description TFIDF 1/3')
    train['desc'] = train['title'].fillna('') + ' ' + train['description'].fillna('')
    test['desc'] = test['title'].fillna('') + ' ' + test['description'].fillna('')
    print_step('Description TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=300000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train['description'].fillna(''))
    print(tfidf_train.shape)
    print_step('Description TFIDF 3/3')
    tfidf_test = tfidf.transform(test['description'].fillna(''))
    print(tfidf_test.shape)

    print_step('Description SVD 1/4')
    svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Description SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Description SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Description SVD 4/4')
    train_svd.columns = ['svd_description_'+str(i+1) for i in range(NCOMP)]
    test_svd.columns = ['svd_description_'+str(i+1) for i in range(NCOMP)]
    train = pd.concat([train, train_svd], axis=1)
    test = pd.concat([test, test_svd], axis=1)

    if not is_in_cache('text_tfidf'):
        print_step('Text TFIDF 1/2')
        tfidf = TfidfVectorizer(ngram_range=(1, 2),
                                max_features=300000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                encoding='KOI8-R')
        tfidf_train = tfidf.fit_transform(train['desc'])
        print(tfidf_train.shape)
        print_step('Text TFIDF 2/2')
        tfidf_test = tfidf.transform(test['desc'])
        print(tfidf_test.shape)
        print_step('Saving to cache...')
        save_in_cache('text_tfidf', tfidf_train, tfidf_test)
    else:
        print_step('Loading from cache...')
        tfidf_train, tfidf_test = load_cache('text_tfidf')

    results = run_cv_model(tfidf_train, tfidf_test, target, runRidge8, rmse, 'desc-ridge')
    train['desc_ridge'] = results['train']
    test['desc_ridge'] = results['test']

    if not is_in_cache('text_char_tfidf'):
        print('~~~~~~~~~~~~~~~~~~~~~~~~')
        print_step('Text Char TFIDF 1/2')
        # Using char n-grams ends up being surprisingly good, HT https://www.kaggle.com/c/avito-demand-prediction/discussion/56061#325063
        tfidf = TfidfVectorizer(ngram_range=(2, 5),
                                max_features=100000,
                                min_df=2,
                                max_df=0.8,
                                binary=True,
                                analyzer='char',
                                encoding='KOI8-R')
        tfidf_train = tfidf.fit_transform(train['desc'])
        print(tfidf_train.shape)
        print_step('Text Char TFIDF 2/2')
        tfidf_test = tfidf.transform(test['desc'])
        print(tfidf_test.shape)
        print_step('Saving to cache...')
        save_in_cache('text_char_tfidf', tfidf_train, tfidf_test)
    else:
        print_step('Loading from cache...')
        tfidf_train, tfidf_test = load_cache('text_char_tfidf')

    results = run_cv_model(tfidf_train, tfidf_test, target, runRidge8, rmse, 'desc-char-ridge')
    train['desc_char_ridge'] = results['train']
    test['desc_char_ridge'] = results['test']

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    train.drop([c for c in train.columns if 'svd' not in c and 'ridge' not in c], axis=1, inplace=True)
    test.drop([c for c in test.columns if 'svd' not in c and 'ridge' not in c], axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('tfidf_ridges', train, test)


print_step('Importing Data 1/5')
tfidf_train, tfidf_test = load_cache('titlecat_tfidf')

print_step('Importing Data 2/5')
tfidf_train2, tfidf_test2 = load_cache('text_tfidf')

print_step('Importing Data 3/5')
tfidf_train3, tfidf_test3 = load_cache('text_char_tfidf')

print_step('Importing Data 4/5')
train = hstack((tfidf_train, tfidf_train2, tfidf_train3)).tocsr()
print_step('Importing Data 5/5')
test = hstack((tfidf_test, tfidf_test2, tfidf_test3)).tocsr()
print(train.shape)
print(test.shape)

if not is_in_cache('full_text_ridge'):
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Run Full Text Ridge')
    results = run_cv_model(train, test, target, runRidge8, rmse, 'text-ridge')
    import pdb
    pdb.set_trace()

    print('~~~~~~~~~~')
    print_step('Cache')
    save_in_cache('full_text_ridge', pd.DataFrame({'full_text_ridge': results['train']}),
                                     pd.DataFrame({'full_text_ridge': results['test']}))

if not is_in_cache('complete_ridge'):
    if not is_in_cache('complete_ridge_data'):
        print('~~~~~~~~~~~~~~~~~~~~~~~')
        print_step('Importing Data 1/5 1/3')
        train_ohe, test_ohe = load_cache('ohe_data')

        print_step('Importing Data 1/5 2/3')
        train = hstack((train, train_ohe)).tocsr()
        print_step('Importing Data 1/5 3/3')
        test = hstack((test, test_ohe)).tocsr()
        print(train.shape)
        print(test.shape)

        print_step('Importing Data 2/5 1/5')
        train_img, test_img = load_cache('img_data')
        print_step('Importing Data 2/5 2/5')
        drops = ['item_id', 'img_path', 'img_std_color', 'img_sum_color', 'img_rms_color',
                 'img_var_color', 'img_average_color', 'deal_probability']
        drops += [c for c in train_img if 'hist' in c]
        dummy_cols = ['img_average_color']
        numeric_cols = list(set(train_img.columns) - set(drops) - set(dummy_cols))
        train_img = train_img[numeric_cols + dummy_cols].fillna(0)
        test_img = test_img[numeric_cols + dummy_cols].fillna(0)
        print_step('Importing Data 2/5 3/5')
        train_img, test_img = bin_and_ohe_data(train_img, test_img,
                                               numeric_cols=numeric_cols,
                                               dummy_cols=dummy_cols)
        print_step('Importing Data 2/5 4/5')
        train = hstack((train, train_img)).tocsr()
        print(train.shape)
        print_step('Importing Data 2/5 5/5')
        test = hstack((test, test_img)).tocsr()
        print(test.shape)

        print_step('GC')
        del test_img
        del train_img
        del tfidf_test
        del tfidf_test2
        del tfidf_test3
        del tfidf_train
        del tfidf_train2
        del tfidf_train3
        del train_ohe
        del test_ohe
        gc.collect()

        print_step('Importing Data 3/5 1/8')
        train_, test_ = get_data()
        train_['city'] = train_['city'] + ', ' + train_['region']
        test_['city'] = test_['city'] + ', ' + test_['region']
        print_step('Importing Data 3/5 2/8')
        # HT: https://www.kaggle.com/jpmiller/russian-cities/data
        # HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
        locations = pd.read_csv('city_latlons.csv')
        print_step('Importing Data 3/5 3/8')
        train_ = train_.merge(locations, how='left', left_on='city', right_on='location')
        print_step('Importing Data 3/5 4/8')
        test_ = test_.merge(locations, how='left', left_on='city', right_on='location')
        print_step('Importing Data 3/5 5/8')
        train_ = train_[['lon', 'lat']]
        test_ = test_[['lon', 'lat']]
        print_step('Importing Data 3/5 6/8')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=['lon', 'lat'],
                                         dummy_cols=[])
        print_step('Importing Data 3/5 7/8')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 3/5 8/8')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Importing Data 4/5 1/8')
        train_, test_ = get_data()
        print_step('Importing Data 4/5 2/8')
        region_macro = pd.read_csv('region_macro.csv')
        print_step('Importing Data 4/5 3/8')
        train_ = train_.merge(region_macro, how='left', on='region')
        print_step('Importing Data 4/5 4/8')
        test_ = test_.merge(region_macro, how='left', on='region')
        print_step('Importing Data 4/5 5/8')
        cols = ['unemployment_rate', 'GDP_PC_PPP', 'HDI']
        train_ = train_[cols]
        test_ = test_[cols]
        print_step('Importing Data 4/5 6/8')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=cols,
                                         dummy_cols=[])
        print_step('Importing Data 4/5 7/8')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 4/5 8/8')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Importing Data 5/5 1/4')
        train_, test_ = load_cache('active_feats')
        train_.fillna(0, inplace=True)
        test_.fillna(0, inplace=True)
        train_.drop('user_id', axis=1, inplace=True)
        test_.drop('user_id', axis=1, inplace=True)
        print_step('Importing Data 5/5 2/4')
        train_, test_ = bin_and_ohe_data(train_, test_,
                                         numeric_cols=train_.columns.values.tolist(),
                                         dummy_cols=[])
        print_step('Importing Data 5/5 3/4')
        train = hstack((train, train_)).tocsr()
        print(train.shape)
        print_step('Importing Data 5/5 4/4')
        test = hstack((test, test_)).tocsr()
        print(test.shape)

        print_step('Caching')
        save_in_cache('complete_ridge_data', train, test)
    else:
        train, test = load_cache('complete_ridge_data')

    print_step('Run Complete Ridge')
    results = run_cv_model(train, test, target, runRidge8, rmse, 'complete-ridge')
    import pdb
    pdb.set_trace()

    print('~~~~~~~~~~')
    print_step('Cache')
    save_in_cache('complete_ridge', pd.DataFrame({'complete_ridge': results['train']}),
                                    pd.DataFrame({'complete_ridge': results['test']}))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Prepping submission file')
    submission = pd.DataFrame()
    submission['item_id'] = test_id
    submission['deal_probability'] = results['test'].clip(0.0, 1.0)
    submission.to_csv('submit/submit_complete_ridge.csv', index=False)
    print_step('Done!')

# [2018-06-05 21:47:05.054174] complete-ridge cv scores : [0.22481497797006353, 0.22409152529039147, 0.22410818210527936, 0.22406284281957914, 0.22472168320897423]
# [2018-06-05 21:47:05.054259] complete-ridge mean cv score : 0.22435984227885752
# [2018-06-05 21:47:05.054364] complete-ridge std cv score : 0.00033514560542958886
