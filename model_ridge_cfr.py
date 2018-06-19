# Ridge model with co-occurent features

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import gc
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from multiprocessing import Pool

# Input paths
train_file = 'train.csv'
test_file = 'test.csv'

output_train_file = 'cache/train_ryan_ridge_sgd_v3.csv'
output_test_file = 'cache/test_ryan_ridge_sgd_v3.csv'

# Defs
def get_cooc_terms(ls1, ls2):
    try:
        return ' '.join([str(a) +'X'+ str(b) for a in ls1 for b in ls2])
    except: print(ls1, ls2)

def create_ridge_features(train, validation, test, model_type, column, ngram_range=2):
    if model_type == 'count':
        
        model = CountVectorizer(ngram_range=(1, ngram_range), max_features=300000,
                                preprocessor=CountVectorizer().build_preprocessor())
        
    elif model_type == 'tfidf':
        
        model = TfidfVectorizer(
                      ngram_range=(1, ngram_range),
                      max_features=300000,
                      norm="l2",
                      strip_accents="unicode",
                      analyzer="word",
                      token_pattern=r"\w{1,}",
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      dtype=np.int16,
                      preprocessor=CountVectorizer().build_preprocessor()
                    )
        

    # Train
    print('fit train ... ')
    X_train = model.fit_transform(train[column])
    X_train = X_train.astype(np.float16)

    # Validation
    print('fit validation ... ')
    X_val = model.transform(validation[column])
    X_val = X_val.astype(np.float16)

    # Test
    print('fit test ... ')
    X_test = model.transform(test[column])
    X_test = X_test.astype(np.float16)

    return X_train, X_val, X_test

def create_df_cols(df):

    df['adjusted_seq_num'] = df['item_seq_number'] - df.groupby('user_id')['item_seq_number'].transform('min')
    df['adjusted_seq_num'] = df.adjusted_seq_num.astype(str)
    df['title'] = df.title.fillna('none')
    df['description'] = df.description.fillna('none')
    df['parent_category_name'] = df.parent_category_name.fillna('none')
    df['category_name'] = df.category_name.fillna('none')
    df['param_1'] = df.param_1.fillna('none')
    df['param_2'] = df.param_2.fillna('none')
    df['param_3'] = df.param_3.fillna('none')
    df['param'] = df['param_1'] + ' ' + df['param_2'] + ' ' + df['param_3']
    df['image_top_1'] = df.price.fillna(-1.).apply(str)
    df['item_seq_number'] = df.item_seq_number.fillna(-1.).apply(str)
    df['price'] = df.price.astype(str)

    # cooc_terms
    df['cat_name_and_title'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.category_name.str.split(' ').values,
                                                                    df.title.str.split(' ').values)))

    df['cat_name_and_description'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.category_name.str.split(' ').values,
                                                                    df.description.str.split(' ').values)))

    df['param_and_description'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.param.str.split(' ').values,
                                                                    df.description.str.split(' ').values)))

    df['param_and_title'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.param.str.split(' ').values,
                                                                    df.title.str.split(' ').values)))

    df['city_and_description'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.city.str.split(' ').values,
                                                                    df.description.str.split(' ').values)))

    df['city_and_title'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.city.str.split(' ').values,
                                                                    df.title.str.split(' ').values)))
    
    df['region_and_title'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.region.str.split(' ').values,
                                                                    df.title.str.split(' ').values)))
    
    df['region_and_desc'] = (list(map(lambda lst1, lst2: get_cooc_terms(lst1, lst2),
                                                                    df.region.str.split(' ').values,
                                                                    df.description.str.split(' ').values)))
    
    df['category_name'] = df.category_name.apply(str)
    df['title'] = df.title.apply(str)
    
    return df

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2017)

# Load train data
train = pd.read_csv(train_file)

test = pd.read_csv(test_file)
y = train.deal_probability
train.drop(['deal_probability'], axis=1, inplace=True)

# Get item id
train_item_id = pd.DataFrame(train['item_id'])
test_item_id = pd.DataFrame(test['item_id'])

# get train and test region and city dummies
nrows = train.shape[0]
cols = ['city', 'region']
df = pd.concat([train[cols], test[cols]])

X_city_and_region = scipy.sparse.csr_matrix(pd.get_dummies(df[['city', 'region']],
                                                           sparse=True)).astype(np.float16)

X_city_and_region_train = X_city_and_region[:nrows, :]
X_city_and_region_test = X_city_and_region[nrows:, :]
del X_city_and_region, df
gc.collect()

# Create train/test features
train = create_df_cols(train)
test = create_df_cols(test)

# Define transformations
cols = [
        ('count', 'adjusted_seq_num', 1),
        ('count', 'image_top_1', 1),
        ('count', 'title', 2),
        ('tfidf', 'cat_name_and_description', 1),
        ('tfidf', 'cat_name_and_title', 1),
        ('tfidf', 'param_and_description', 1),
        ('tfidf', 'param_and_title', 1),
        ('tfidf', 'city_and_description', 1),
        ('tfidf', 'city_and_title', 1),
        ('tfidf', 'description', 3),
        ('tfidf', 'parent_category_name', 2),
        ('tfidf', 'category_name', 2),
        ('tfidf', 'param', 2),
       ]

# Create Ridge & SGD Model

oof_predictions = np.zeros((train.shape[0], 2))
test_predictions = np.zeros((test.shape[0], 2))

for train_index, test_index in kf.split(train):
    
    X_train = []
    X_valid = []
    X_test = []
    
    train_X, valid_X = train.iloc[train_index, :], train.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train.append(X_city_and_region_train[train_index])
    X_valid.append(X_city_and_region_train[test_index])
    X_test.append(X_city_and_region_test)
    
    for f in cols:
        print(f)
        train_f, valid_f, test_f = create_ridge_features(train_X, valid_X, test, 
                                                         f[0], f[1], f[2])
        X_train.append(train_f)
        X_valid.append(valid_f)
        X_test.append(test_f)
        
        train_X.drop(f[1], axis=1, inplace=True)
        valid_X.drop(f[1], axis=1, inplace=True)
        
    train_cv = hstack(X_train)
    valid_cv = hstack(X_valid)
    test_cv = hstack(X_test)
    
    # Ridge Model
    model = Ridge(alpha=30.0, copy_X=True, fit_intercept=True,
                  max_iter=20.0, normalize=False, random_state=101,
                  solver='sag', tol=0.03)

    print("Fitting Model - Ridge")

    model.fit(train_cv, y_train)
    preds = model.predict(valid_cv)
    test_preds = model.predict(test_cv)

    print('Ridge RMSE = ', np.sqrt(mean_squared_error(y_test, preds)))
    oof_predictions[test_index, 0] = np.clip(preds, 0.0, 1.0)
    test_predictions[:, 0] += test_preds / 5.0
    
    # SGD Model
    model = SGDRegressor(eta0=0.2, fit_intercept=True, power_t=0.229, alpha=0.0,
                         random_state=101, max_iter=100, tol=1e-3)
    
    print("Fitting Model - SGD")

    model.fit(train_cv, y_train)
    preds = model.predict(valid_cv)
    test_preds = model.predict(test_cv)

    print('SGD RMSE = ', np.sqrt(mean_squared_error(y_test, preds)))
    oof_predictions[test_index, 1] = np.clip(preds, 0.0, 1.0)
    test_predictions[:, 1] += test_preds / 5.0

print('Ridge RMSE = ', np.sqrt(mean_squared_error(y, oof_predictions[:, 0])))
print('SGD RMSE = ', np.sqrt(mean_squared_error(y, oof_predictions[:, 1])))

train_item_id['oof_ridge'] = oof_predictions[:, 0]
train_item_id['oof_sgd'] = oof_predictions[:, 1]

test_item_id['oof_ridge'] = test_predictions[:, 0]
test_item_id['oof_sgd'] = test_predictions[:, 1]

train_item_id.to_csv(output_train_file, index=False, header=True)
test_item_id.to_csv(output_test_file, index=False, header=True)
