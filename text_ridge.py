import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from cv import run_cv_model
from utils import print_step, rmse
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runRidge(train_X, train_y, test_X, test_y, test_X2):
    model = Ridge()
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data')
train, test = get_data()
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']


if not is_in_cache('tfidf'):
    print('~~~~~~~~~~~~')
    print_step('Merging')
    merge = pd.concat([train, test])
    cat_feats = ['region', 'city', 'parent_category_name', 'category_name', 'param_1',
                 'param_2', 'param_3', 'user_type', 'image_top_1']
    num_feats = ['price', 'item_seq_number']

    print('~~~~~~~~~~~~~~~')
    print_step('Impute 1/5')
    merge['param_1'].fillna('missing', inplace=True)
    print_step('Impute 2/5')
    merge['param_2'].fillna('missing', inplace=True)
    print_step('Impute 3/5')
    merge['param_3'].fillna('missing', inplace=True)
    print_step('Impute 4/5')
    merge['price'].fillna(merge['price'].median(), inplace=True)
    print_step('Impute 5/5')
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')

    print('~~~~~~~~~~~~')
    print_step('Munging')
    merge['text'] = merge['title'] + ' ' + merge['description'].fillna('')
    merge['price'] = merge['price'].apply(np.log1p)


    print('~~~~~~~~~~')
    print_step('TFIDF')
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=100000,
                            min_df=2,
                            max_df=0.8,
                            binary=True,
                            encoding='KOI8-R')
    tfidf_merge = tfidf.fit_transform(merge['text'])
    print(tfidf_merge.shape)

    print('~~~~~~~~~~~~~~~~')
    print_step('Dummies 1/3')
    print(merge.shape)
    merge = merge[cat_feats + num_feats]
    print('-')
    print(merge.shape)

    print_step('Dummies 2/3')
    for col in cat_feats:
        le = LabelEncoder()
        merge[col] = le.fit_transform(merge[col])

    print_step('Dummies 3/3')
    ohe = OneHotEncoder(categorical_features=[merge.columns.get_loc(c) for c in cat_feats])
    merge = ohe.fit_transform(merge)

    print_step('Combine')
    merge = hstack((merge, tfidf_merge)).tocsr()

    print_step('Unmerge')
    dim = train.shape[0]
    train = merge[:dim]
    test = merge[dim:]

    print_step('Saving in cache')
    save_in_cache('tfidf', train, test)
else:
    train, test = load_cache('tfidf')
    print(train.shape)
    print(test.shape)


print('~~~~~~~~~~~~~~')
print_step('Run Ridge')
results = run_cv_model(train, test, target, runRidge, rmse, 'ridge') # ~9m19s per fold?
# Score ~0.236 submit, so 0.2312-0.2339 CV
import pdb
pdb.set_trace()

#print('~~~~~~~~~~')
#print_step('Cache')
#save_in_cache('lvl1_lgb', train, test)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_lgb.csv', index=False)
print_step('Done!')
