import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import rmse, print_step
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
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Title SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Title SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Title SVD 4/4')
    train_svd.columns = ['svd_title_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_title_'+str(i+1) for i in range(10)]
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
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Titlecat SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Titlecat SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Titlecat SVD 4/4')
    train_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_titlecat_'+str(i+1) for i in range(10)]
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
    svd = TruncatedSVD(n_components=10, algorithm='arpack')
    svd.fit(tfidf_train)
    print_step('Description SVD 2/4')
    train_svd = pd.DataFrame(svd.transform(tfidf_train))
    print_step('Description SVD 3/4')
    test_svd = pd.DataFrame(svd.transform(tfidf_test))
    print_step('Description SVD 4/4')
    train_svd.columns = ['svd_description_'+str(i+1) for i in range(10)]
    test_svd.columns = ['svd_description_'+str(i+1) for i in range(10)]
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


print('~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/6')
train_fe, test_fe = load_cache('ohe_data')

print_step('Importing Data 2/6')
tfidf_train, tfidf_test = load_cache('titlecat_tfidf')

print_step('Importing Data 3/6')
tfidf_train2, tfidf_test2 = load_cache('text_tfidf')

print_step('Importing Data 4/6')
tfidf_train3, tfidf_test3 = load_cache('text_char_tfidf')

print_step('Importing Data 5/6')
train = hstack((tfidf_train, tfidf_train2, tfidf_train3)).tocsr()
print_step('Importing Data 6/6')
test = hstack((tfidf_test, tfidf_test2, tfidf_test3)).tocsr()
print(train.shape)
print(test.shape)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Run Full Text Ridge')
results = run_cv_model(train, test, target, runRidge8, rmse, 'text-ridge')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('full_text_ridge', pd.DataFrame({'full_text_ridge': results['train']}),
							     pd.DataFrame({'full_text_ridge': results['test']}))
