from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


def runLGB(train_X, train_y, test_X, test_y, test_X2):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.03,
              'application': 'regression',
              'num_leaves': 250,
              'verbosity': -1,
              'metric': 'rmse',
              'data_random_seed': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.1,
              'nthread': 16, #max(mp.cpu_count() - 2, 2),
              'lambda_l1': 6,
              'lambda_l2': 6,
              'min_data_in_leaf': 40}
    print_step('Train LGB')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=1800,
                      valid_sets=watchlist,
                      verbose_eval=100)
    print_step('Feature importance')
    pprint(sorted(list(zip(model.feature_importance(), train_X.columns)), reverse=True))
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/15')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

# print('~~~~~~~~~~~~~~~~~~~~~~~~')
# print_step('Importing Data 2/15')
# train_fe, test_fe = load_cache('data_with_fe')

# print_step('Importing Data 3/15 1/4')
# train_img, test_img = load_cache('img_data')
# print_step('Importing Data 3/15 2/4')
# cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
# train_img = train_img[cols].fillna(0)
# test_img = test_img[cols].fillna(0)
# print_step('Importing Data 3/15 3/4')
# train_fe = pd.concat([train_fe, train_img], axis=1)
# print_step('Importing Data 3/15 4/4')
# test_fe = pd.concat([test_fe, test_img], axis=1)

# print_step('Importing Data 4/15 1/4')
# # HT: https://www.kaggle.com/jpmiller/russian-cities/data
# # HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
# locations = pd.read_csv('city_latlons.csv')
# print_step('Importing Data 4/15 2/4')
# train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
# print_step('Importing Data 4/15 3/4')
# test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
# print_step('Importing Data 4/15 4/4')
# train_fe.drop('location', axis=1, inplace=True)
# test_fe.drop('location', axis=1, inplace=True)

# print_step('Importing Data 5/15 1/3')
# region_macro = pd.read_csv('region_macro.csv')
# print_step('Importing Data 5/15 2/3')
# train_fe = train_fe.merge(region_macro, how='left', on='region')
# print_step('Importing Data 5/15 3/3')
# test_fe = test_fe.merge(region_macro, how='left', on='region')


# # print_step('Importing Data 6/15 1/3')
# # train_target_encoding, test_target_encoding = load_cache('target_encoding')
# # print_step('Importing Data 6/15 2/3')
# # train_fe = pd.concat([train_fe, train_target_encoding], axis=1)
# # print_step('Importing Data 6/15 3/3')
# # test_fe = pd.concat([test_fe, test_target_encoding], axis=1)
# print_step('Importing Data 6/15 1/3')
# train_entity, test_entity = load_cache('price_entity_embed')
# print_step('Importing Data 6/15 2/3')
# train_fe = pd.concat([train_fe, train_entity], axis=1)
# print_step('Importing Data 6/15 3/3')
# test_fe = pd.concat([test_fe, test_entity], axis=1)


# print_step('Importing Data 6/15 1/5')
# train_active_feats, test_active_feats = load_cache('active_feats')
# train_active_feats.fillna(0, inplace=True)
# test_active_feats.fillna(0, inplace=True)
# print_step('Importing Data 6/15 2/5')
# train_fe = pd.concat([train_fe, train_active_feats], axis=1)
# print_step('Importing Data 6/15 3/5')
# test_fe = pd.concat([test_fe, test_active_feats], axis=1)
# print_step('Importing Data 6/15 4/5')
# train_fe['user_items_per_day'] = train_fe['n_user_items'] / train_fe['user_num_days']
# test_fe['user_items_per_day'] = test_fe['n_user_items'] / test_fe['user_num_days']
# train_fe['img_size_ratio'] = train_fe['img_file_size'] / (train_fe['img_size_x'] * train_fe['img_size_y'])
# test_fe['img_size_ratio'] = test_fe['img_file_size'] / (test_fe['img_size_x'] * test_fe['img_size_y'])
# print_step('Importing Data 6/15 5/5')
# train_fe.drop('user_id', axis=1, inplace=True)
# test_fe.drop('user_id', axis=1, inplace=True)


# print_step('Importing Data 7/15 1/8')
# train, test = get_data()
# print_step('Importing Data 7/15 2/8')
# train_nima, test_nima = load_cache('img_nima')
# print_step('Importing Data 7/15 3/8')
# train = train.merge(train_nima, on = 'image', how = 'left')
# print_step('Importing Data 7/15 4/8')
# test = test.merge(test_nima, on = 'image', how = 'left')
# print_step('Importing Data 7/15 5/8')
# cols = ['mobile_mean', 'mobile_std', 'inception_mean', 'inception_std',
# 		'nasnet_mean', 'nasnet_std']
# train_fe[cols] = train[cols].fillna(0)
# test_fe[cols] = test[cols].fillna(0)
# print_step('Importing Data 7/15 6/8')
# train_nima, test_nima = load_cache('img_nima_softmax')
# print_step('Importing Data 7/15 7/8')
# train = train.merge(train_nima, on = 'image', how = 'left')
# test = test.merge(test_nima, on = 'image', how = 'left')
# print_step('Importing Data 7/15 8/8')
# cols = [x + '_' + str(y) for x in ['mobile', 'inception', 'nasnet'] for y in range(1, 11)]
# train_fe[cols] = train[cols].fillna(0)
# test_fe[cols] = test[cols].fillna(0)
# del train, test, train_nima, test_nima


# print_step('Importing Data 8/15 1/2')
# train_ridge, test_ridge = load_cache('tfidf_ridges')
# print_step('Importing Data 8/15 2/2')
# cols = [c for c in train_ridge.columns if 'svd' in c or 'tfidf' in c]
# train_fe[cols] = train_ridge[cols]
# test_fe[cols] = test_ridge[cols]


# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print_step('Converting to category')
# train_fe['image_top_1'] = train_fe['image_top_1'].astype('str').fillna('missing')
# test_fe['image_top_1'] = test_fe['image_top_1'].astype('str').fillna('missing')
# cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
#             'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'day_of_week',
#             'img_average_color']
# for col in train_fe.columns:
#     print(col)
#     if col in cat_cols:
#         train_fe[col] = train_fe[col].astype('category')
#         test_fe[col] = test_fe[col].astype('category')
#     else:
#         train_fe[col] = train_fe[col].astype(np.float64)
#         test_fe[col] = test_fe[col].astype(np.float64)


print('~~~~~~~~~~~~~~~~~~~~~')
print_step('Adding Embedding')
EMBEDDING_FILE = 'cache/avito_fasttext_300d.txt'

print_step('Embedding 1/7')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

def text_to_embedding(text):
    mean = np.mean([embeddings_index.get(w, np.zeros(embed_size)) for w in text.split()], axis=0)
    if mean.shape == ():
        return np.zeros(embed_size)
    else:
        return mean

print_step('Embedding 2/7')
train_embeddings = (train['title'].str.cat([
                        df['description'],
                    ], sep=' ', na_rep='')
                    .astype(str)
                    .fillna('missing')
                    .apply(clean_text)
                    .apply(text_to_embedding))

print_step('Embedding 3/7')
test_embeddings = (test['title'].str.cat([
                        df['description'],
                    ], sep=' ', na_rep='')
                    .astype(str)
                    .fillna('missing')
                    .apply(clean_text)
                    .apply(text_to_embedding))

print_step('Embedding 4/7')
train_embeddings_df = pd.DataFrame(train_embeddings.values.tolist(),
                                   columns = ['embed' + str(i) for i in range(embed_size)])
print_step('Embedding 5/7')
test_embeddings_df = pd.DataFrame(test_embeddings.values.tolist(),
                                  columns = ['embed' + str(i) for i in range(embed_size)])

print_step('Embedding 6/7')
train = pd.concat([train, train_embeddings_df], axis=1)
print_step('Embedding 7/7')
test = pd.concat([test, test_embeddings_df], axis=1)
import pdb
pdb.set_trace()


print('~~~~~~~~~~~~')
print_step('Run LGB')
print(train_fe.shape)
print(test_fe.shape)
results = run_cv_model(train_fe, test_fe, target, runLGB, rmse, 'embedding_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('embedding_lgb', pd.DataFrame({'embedding_lgb': results['train']}),
                               pd.DataFrame({'embedding_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_embedding_lgb.csv', index=False)
