from pprint import pprint

import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, normalize_text
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
              'lambda_l2': 1}
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

if not is_in_cache('ohe_data'):
    print('~~~~~~~~~~~~')
    print_step('Merging')
    merge = pd.concat([train, test])

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Titlecat TFIDF 1/3')
    train['titlecat'] = train['parent_category_name'] + ' ' + train['category_name'] + ' ' + train['param_1'].fillna('') + ' ' + train['param_2'].fillna('') + ' ' + train['param_3'].fillna('') + ' ' + train['title']
    test['titlecat'] = test['parent_category_name'] + ' ' + test['category_name'] + ' ' + test['param_1'].fillna('') + ' ' + test['param_2'].fillna('') + ' ' + test['param_3'].fillna('') + ' ' + test['title']
    print_step('Titlecat TFIDF 2/3')
    tfidf = TfidfVectorizer(ngram_range=(1, 1),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train = tfidf.fit_transform(train['titlecat'])
    print(tfidf_train.shape)
    print_step('Titlecat TFIDF 3/3')
    tfidf_test = tfidf.transform(test['titlecat'])
    print(tfidf_test.shape)

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    if not is_in_cache('normalized_desc'):
        print_step('Normalize text 1/3')
        train['desc'] = train['title'] + ' ' + train['description'].fillna('')
        test['desc'] = test['title'] + ' ' + test['description'].fillna('')
        print_step('Normalize text 2/3')
        # HT @IggiSv9t https://www.kaggle.com/iggisv9t/handling-russian-language-inflectional-structure
        train['desc'] = train['desc'].astype(str).apply(normalize_text)
        print_step('Normalize text 3/3')
        test['desc'] = test['desc'].astype(str).apply(normalize_text)
    else:
        print_step('Loading normalized data from cache')
        normalized_desc_train, normalized_desc_test = load_cache('normalized_desc')
        train['desc'] = normalized_desc_train['desc'].fillna('')
        test['desc'] = normalized_desc_test['desc'].fillna('')

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Text TFIDF 1/2')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_train2 = tfidf.fit_transform(train['desc'])
    print(tfidf_train2.shape)
    print_step('Text TFIDF 2/2')
    tfidf_test2 = tfidf.transform(test['desc'])
    print(tfidf_test2.shape)

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    print(train.shape)
    print(test.shape)
    drops = ['activation_date', 'description', 'title', 'desc', 'titlecat', 'image', 'user_id']
    train.drop(drops, axis=1, inplace=True)
    test.drop(drops, axis=1, inplace=True)
    print('-')
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~~')
    print_step('Merging')
    merge = pd.concat([train, test])
    print(merge.shape)

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Imputation 1/6')
    merge['param_1'].fillna('missing', inplace=True)
    print_step('Imputation 2/6')
    merge['param_2'].fillna('missing', inplace=True)
    print_step('Imputation 3/6')
    merge['param_3'].fillna('missing', inplace=True)
    print_step('Imputation 4/6')
    merge['price_missing'] = merge['price'].isna().astype(int)
    merge['price'].fillna(merge['price'].median(), inplace=True)
    print_step('Imputation 5/6')
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')
    print_step('Imputation 6/6')
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    merge['city'] = merge['city'] + '_' + merge['region']
    print(merge.columns)
    print(merge.shape)

    print('~~~~~~~~~~~~~~~~')
    print_step('Dummies 1/2')
    dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1',
                  'day_of_week', 'region', 'city', 'param_1', 'param_2', 'param_3']
    for col in dummy_cols:
        le = LabelEncoder()
        merge[col] = le.fit_transform(merge[col])
    print_step('Dummies 2/2')
    ohe = OneHotEncoder(categorical_features=[merge.columns.get_loc(c) for c in dummy_cols])
    merge = ohe.fit_transform(merge)
    print(merge.shape)

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    merge = merge.tocsr()
    dim = train.shape[0]
    train = merge[:dim]
    test = merge[dim:]
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~~')
    print_step('Combine')
    # Just jam the TFIDF matrix into the features, cast to sparse matrix
    # and let LGB handle it. This has mostly worked fine in previous competitions.
    train = hstack((train, tfidf_train, tfidf_train2)).tocsr()
    test = hstack((test, tfidf_test, tfidf_test2)).tocsr()
    print(train.shape)
    print(test.shape)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('ohe_data', train, test)
else:
    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache Loading')
    train, test = load_cache('ohe_data')
    print(train.shape)
    print(test.shape)


print('~~~~~~~~~~~~')
print_step('Run LGB')
results = run_cv_model(train, test, target, runLGB, rmse, 'lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~~~')
print_step('Run LGB')
save_in_cache('deep_text_lgb', pd.DataFrame({'deep_text_lgb': results['train']}),
							   pd.DataFrame({'deep_text_lgb': results['test']}))

#print('~~~~~~~~~~')
#print_step('Cache')
#save_in_cache('lvl1_lgb', train, test)

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
