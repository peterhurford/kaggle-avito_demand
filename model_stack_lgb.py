import gc
from pprint import pprint

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

from cv import run_cv_model
from utils import print_step, rmse, clean_text
from cache import get_data, is_in_cache, load_cache, save_in_cache


params = {'learning_rate': 0.03, #0.06
          'application': 'regression', #'poisson'
		  #'poisson_max_delta_step': 1.5,
          'num_leaves': 250,
          'verbosity': -1,
          'metric': 'rmse',
          'data_random_seed': 3,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.1,
          'nthread': 16, #max(mp.cpu_count() - 2, 2),
          'lambda_l1': 6,
          'lambda_l2': 6,
          'min_data_in_leaf': 40,
          'num_rounds': 5200, #5400 for poisson, 4200 for regression
          'verbose_eval': 100}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print_step('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval)
    print_step('Feature importance')
    pprint(sorted(list(zip(model.feature_importance(), train_X.columns)), reverse=True))
    print_step('Predict 1/2')
    pred_test_y = model.predict(test_X)
    print_step('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2


print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 1/19')
train, test = get_data()

print('~~~~~~~~~~~~~~~')
print_step('Subsetting')
target = train['deal_probability']
train_id = train['item_id']
test_id = test['item_id']
train.drop(['deal_probability', 'item_id'], axis=1, inplace=True)
test.drop(['item_id'], axis=1, inplace=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 2/19')
train_fe, test_fe = load_cache('data_with_fe')

print_step('Importing Data 3/19 1/4')
train_img, test_img = load_cache('img_data')
print_step('Importing Data 3/19 2/4')
cols = ['img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color', 'img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21', 'img_kurtosis', 'img_skew', 'thing1', 'thing2']
train_img = train_img[cols].fillna(0)
test_img = test_img[cols].fillna(0)
print_step('Importing Data 3/19 3/4')
train_fe = pd.concat([train_fe, train_img], axis=1)
print_step('Importing Data 3/19 4/4')
test_fe = pd.concat([test_fe, test_img], axis=1)

print_step('Importing Data 4/19 1/4')
# HT: https://www.kaggle.com/jpmiller/russian-cities/data
# HT: https://www.kaggle.com/jpmiller/exploring-geography-for-1-5m-deals/notebook
locations = pd.read_csv('city_latlons.csv')
print_step('Importing Data 4/19 2/4')
train_fe = train_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 4/19 3/4')
test_fe = test_fe.merge(locations, how='left', left_on='city', right_on='location')
print_step('Importing Data 4/19 4/4')
train_fe.drop('location', axis=1, inplace=True)
test_fe.drop('location', axis=1, inplace=True)

print_step('Importing Data 5/19 1/3')
region_macro = pd.read_csv('region_macro.csv')
print_step('Importing Data 5/19 2/3')
train_fe = train_fe.merge(region_macro, how='left', on='region')
print_step('Importing Data 5/19 3/3')
test_fe = test_fe.merge(region_macro, how='left', on='region')

print_step('Importing Data 6/19 1/3')
train_entity, test_entity = load_cache('price_entity_embed')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_entity], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_entity], axis=1)

print_step('Importing Data 6/19 1/3')
train_ryan, test_ryan = load_cache('ryan_lgbm_features')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_ryan], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_ryan], axis=1)

print_step('Importing Data 6/19 1/3')
train_le, test_le = load_cache('le')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_le], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_le], axis=1)
train_fe.drop('item_id', axis=1, inplace=True)
test_fe.drop('item_id', axis=1, inplace=True)

print_step('Importing Data 6/19 1/3')
train_numeric, test_numeric = load_cache('numeric')
train_numeric.drop(['item_id', 'item_seq_number', 'price_missing'], axis=1, inplace=True)
test_numeric.drop(['item_id', 'item_seq_number', 'price_missing'], axis=1, inplace=True)
train_fe.drop(['image_top_1', 'price'], axis=1, inplace=True)
test_fe.drop(['image_top_1', 'price'], axis=1, inplace=True)
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_numeric], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_numeric], axis=1)

print_step('Importing Data 7/19 1/5')
train_active_feats, test_active_feats = load_cache('active_feats')
train_active_feats.fillna(0, inplace=True)
test_active_feats.fillna(0, inplace=True)
print_step('Importing Data 7/19 2/5')
train_fe = pd.concat([train_fe, train_active_feats], axis=1)
print_step('Importing Data 7/19 3/5')
test_fe = pd.concat([test_fe, test_active_feats], axis=1)
print_step('Importing Data 7/19 4/5')
train_fe['user_items_per_day'] = train_fe['n_user_items'] / train_fe['user_num_days']
test_fe['user_items_per_day'] = test_fe['n_user_items'] / test_fe['user_num_days']
train_fe['img_size_ratio'] = train_fe['img_file_size'] / (train_fe['img_size_x'] * train_fe['img_size_y'])
test_fe['img_size_ratio'] = test_fe['img_file_size'] / (test_fe['img_size_x'] * test_fe['img_size_y'])
print_step('Importing Data 7/19 5/5')
train_fe.drop('user_id', axis=1, inplace=True)
test_fe.drop('user_id', axis=1, inplace=True)

print_step('Importing Data 8/19 1/8')
train, test = get_data()
print_step('Importing Data 8/19 2/8')
train_nima, test_nima = load_cache('img_nima')
print_step('Importing Data 8/19 3/8')
train = train.merge(train_nima, on = 'image', how = 'left')
print_step('Importing Data 8/19 4/8')
test = test.merge(test_nima, on = 'image', how = 'left')
print_step('Importing Data 7/19 5/8')
cols = ['mobile_mean', 'mobile_std', 'inception_mean', 'inception_std',
        'nasnet_mean', 'nasnet_std']
train_fe[cols] = train[cols].fillna(0)
test_fe[cols] = test[cols].fillna(0)
print_step('Importing Data 8/19 6/8')
train_nima, test_nima = load_cache('img_nima_softmax')
print_step('Importing Data 8/19 7/8')
train = train[['item_id', 'image']].merge(train_nima, on = 'image', how = 'left')
test = test[['item_id', 'image']].merge(test_nima, on = 'image', how = 'left')
print_step('Importing Data 8/19 8/8')
cols = [x + '_' + str(y) for x in ['mobile', 'inception', 'nasnet'] for y in range(1, 11)]
train_fe[cols] = train[cols].fillna(0)
test_fe[cols] = test[cols].fillna(0)

print_step('Importing Data 9/19 1/20')
train_nasnet, test_nasnet = load_cache('nasnet')
print_step('Importing Data 9/19 2/20')
train_fe = pd.concat([train_fe, train_nasnet], axis=1)
print_step('Importing Data 9/19 3/20')
test_fe = pd.concat([test_fe, test_nasnet], axis=1)
print_step('Importing Data 9/19 4/20')
train_fe.drop('image', axis=1, inplace=True)
test_fe.drop('image', axis=1, inplace=True)
print_step('Importing Data 9/19 5/20')
train_xception, test_xception = load_cache('xception')
print_step('Importing Data 9/19 6/20')
train_fe = pd.concat([train_fe, train_xception], axis=1)
print_step('Importing Data 9/19 7/20')
test_fe = pd.concat([test_fe, test_xception], axis=1)
print_step('Importing Data 9/19 8/20')
train_fe.drop('image', axis=1, inplace=True)
test_fe.drop('image', axis=1, inplace=True)
print_step('Importing Data 9/19 9/20')
train_inception_resnet_v2, test_inception_resnet_v2 = load_cache('inception_resnet_v2')
print_step('Importing Data 9/19 10/20')
train_fe = pd.concat([train_fe, train_inception_resnet_v2], axis=1)
print_step('Importing Data 9/19 11/20')
test_fe = pd.concat([test_fe, test_inception_resnet_v2], axis=1)
print_step('Importing Data 9/19 12/20')
train_fe.drop('image', axis=1, inplace=True)
test_fe.drop('image', axis=1, inplace=True)
print_step('Importing Data 9/19 13/20')
del train, test, train_nima, test_nima, train_xception, test_xception
del train_nasnet, test_nasnet, train_inception_resnet_v2, test_inception_resnet_v2
gc.collect()
print_step('Importing Data 9/19 14/20')
train_fe['img_xception_inception_agree'] = (train_fe['xception_top_1'] == train_fe['inception_resnet_v2_top_1']).astype(int)
test_fe['img_xception_inception_agree'] = (test_fe['xception_top_1'] == test_fe['inception_resnet_v2_top_1']).astype(int)
print_step('Importing Data 9/19 15/20')
train_fe['img_nasnet_inception_agree'] = (train_fe['nasnet_top_1'] == train_fe['inception_resnet_v2_top_1']).astype(int)
test_fe['img_nasnet_inception_agree'] = (test_fe['nasnet_top_1'] == test_fe['inception_resnet_v2_top_1']).astype(int)
print_step('Importing Data 9/19 16/20')
train_fe['img_xception_nasnet_agree'] = (train_fe['xception_top_1'] == train_fe['nasnet_top_1']).astype(int)
test_fe['img_xception_nasnet_agree'] = (test_fe['xception_top_1'] == test_fe['nasnet_top_1']).astype(int)
print_step('Importing Data 9/19 17/20')
train_fe['img_all_agree'] = (train_fe['img_xception_nasnet_agree'] & train_fe['img_nasnet_inception_agree'] & train_fe['img_xception_inception_agree']).astype(int)
test_fe['img_all_agree'] = (test_fe['img_xception_nasnet_agree'] & test_fe['img_nasnet_inception_agree'] & test_fe['img_xception_inception_agree']).astype(int)
print_step('Importing Data 9/19 18/20')
train_fe['img_mean_label_score'] = np.mean([train_fe['xception_prob'], train_fe['inception_resnet_v2_prob'], train_fe['nasnet_prob']], axis=0)
test_fe['img_mean_label_score'] = np.mean([test_fe['xception_prob'], test_fe['inception_resnet_v2_prob'], test_fe['nasnet_prob']], axis=0)
print_step('Importing Data 9/19 19/20')
train_fe['img_std_label_score'] = np.std([train_fe['xception_prob'], train_fe['inception_resnet_v2_prob'], train_fe['nasnet_prob']], axis=0)
test_fe['img_std_label_score'] = np.std([test_fe['xception_prob'], test_fe['inception_resnet_v2_prob'], test_fe['nasnet_prob']], axis=0)
print_step('Importing Data 9/19 20/20')
train_fe.drop(['xception_top_1', 'nasnet_top_1', 'inception_resnet_v2_top_1'], axis=1, inplace=True)
test_fe.drop(['xception_top_1', 'nasnet_top_1', 'inception_resnet_v2_top_1'], axis=1, inplace=True)

train_img = np.load('cache/train_image_meta_wihtout_zero_size.npy')
img_cols = ['lightness_mean_yuv', 'lightness_std_yuv', 'lightness_mean_hsl', 'lightness_std_hsl', 'saturation_mean_hsv',
            'saturation_std_hsv', 'colorfulness', 'gray_std', 'red_mean', 'green_mean', 'blue_mean', 'image_x', 'image_y',
            'image_area', 'blurrness']
train_img = pd.DataFrame(train_img, columns = img_cols)
test_img = np.load('cache/test_image_meta.npy')
test_img = pd.DataFrame(test_img, columns = img_cols)
train, test = get_data()
drops = ['red_mean', 'green_mean', 'blue_mean', 'image_x', 'image_y', 'image_area', 'blurrness']
img_cols = list(set(img_cols) - set(drops))
train_img.drop(drops, axis=1, inplace=True)
test_img.drop(drops, axis=1, inplace=True)
train_fe['image'] = train['image']
test_fe['image'] = test['image']
for col in img_cols:
  train_fe[col] = 0
  test_fe[col] = 0
train_fe.loc[train_fe['image'].notnull(), img_cols] = train_img
test_fe.loc[test_fe['image'].notnull(), img_cols] = test_img
train_fe.drop('image', inplace=True, axis=1)
test_fe.drop('image', inplace=True, axis=1)


print_step('Importing Data 10/19 1/2')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 10/19 2/2')
cols = [c for c in train_ridge.columns if 'svd' in c or 'tfidf' in c]
train_fe[cols] = train_ridge[cols]
test_fe[cols] = test_ridge[cols]


EMBEDDING_FILE = 'cache/avito_fasttext_300d.txt'
EMBED_SIZE = 300
NCOMP = 20

def text_to_embedding(text):
    mean = np.mean([embeddings_index.get(w, np.zeros(EMBED_SIZE)) for w in text.split()], axis=0)
    if mean.shape == ():
        return np.zeros(EMBED_SIZE)
    else:
        return mean

print_step('Importing Data 11/19 1/3')
if not is_in_cache('avito_fasttext_300d'):
    print_step('Embedding 1/5')
    train, test = get_data()

    print_step('Embedding 1/5')
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

    print_step('Embedding 2/5')
    train_embeddings = (train['title'].str.cat([
                            train['description'],
                        ], sep=' ', na_rep='')
                        .astype(str)
                        .fillna('missing')
                        .apply(clean_text)
                        .apply(text_to_embedding))

    print_step('Embedding 3/5')
    test_embeddings = (test['title'].str.cat([
                            test['description'],
                        ], sep=' ', na_rep='')
                        .astype(str)
                        .fillna('missing')
                        .apply(clean_text)
                        .apply(text_to_embedding))

    print_step('Embedding 4/5')
    train_embeddings_df = pd.DataFrame(train_embeddings.values.tolist(),
                                       columns = ['embed' + str(i) for i in range(EMBED_SIZE)])
    print_step('Embedding 5/5')
    test_embeddings_df = pd.DataFrame(test_embeddings.values.tolist(),
                                      columns = ['embed' + str(i) for i in range(EMBED_SIZE)])
    print_step('Caching...')
    save_in_cache('avito_fasttext_300d', train_embeddings_df, test_embeddings_df)
else:
    train_embeddings_df, test_embeddings_df = load_cache('avito_fasttext_300d')

print_step('Importing Data 11/19 2/3')
train_fe = pd.concat([train_fe, train_embeddings_df], axis=1)
print_step('Importing Data 11/19 3/3')
test_fe = pd.concat([test_fe, test_embeddings_df], axis=1)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Converting to category')
train_fe['image_top_1_imputed'] = train_fe['image_top_1_imputed'].astype('str').fillna('missing')
test_fe['image_top_1_imputed'] = test_fe['image_top_1_imputed'].astype('str').fillna('missing')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'cat_bin',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1_imputed', 'day_of_week', 'img_average_color']
for col in train_fe.columns:
    print(col)
    if col in cat_cols:
        train_fe[col] = train_fe[col].fillna('missing').astype('category')
        test_fe[col] = test_fe[col].fillna('missing').astype('category')
    else:
        train_fe[col] = train_fe[col].fillna(0).astype(np.float64)
        test_fe[col] = test_fe[col].fillna(0).astype(np.float64)


# print('~~~~~~~~~~~~~~~~~~~~~~')
# print_step('Pre-flight checks')
# for col in train_fe.columns:
#     print('##')
#     print(col)
#     print('-')
#     print(train_fe[col].values)
#     print('-')
#     print(test_fe[col].values)
#     print('-')
# print('-')


# print('~~~~~~~~~~~~')
# print_step('Run LGB')
# print(train_fe.shape)
# print(test_fe.shape)
# results = run_cv_model(train_fe, test_fe, target, runLGB, params, rmse, 'base_lgb')
# import pdb
# pdb.set_trace()

# print('~~~~~~~~~~')
# print_step('Cache')
# save_in_cache('base_lgb', pd.DataFrame({'base_lgb': results['train']}),
#                           pd.DataFrame({'base_lgb': results['test']}))

# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print_step('Prepping submission file')
# submission = pd.DataFrame()
# submission['item_id'] = test_id
# submission['deal_probability'] = results['test'].clip(0.0, 1.0)
# submission.to_csv('submit/submit_base_lgb.csv', index=False)


print('~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 12/19 1/7')
train_te = train_fe.copy()
test_te = test_fe.copy()
for col in cat_cols:
    if col in train_te.columns:
        train_te.drop(col, axis=1, inplace=True)
        test_te.drop(col, axis=1, inplace=True)
train_te.drop([c for c in train_te.columns if 'inter' in c], axis=1, inplace=True)
test_te.drop([c for c in test_te.columns if 'inter' in c], axis=1, inplace=True)
print_step('Importing Data 12/19 2/7')
train_target_encoding, test_target_encoding = load_cache('target_encoding_1000')
print_step('Importing Data 12/19 3/7')
train_te = pd.concat([train_te, train_target_encoding], axis=1)
print_step('Importing Data 12/19 4/7')
test_te = pd.concat([test_te, test_target_encoding], axis=1)
print_step('Importing Data 12/19 5/7')
train_nb, test_nb = load_cache('naive_bayes_svd_10')
print_step('Importing Data 12/19 6/7')
train_te = pd.concat([train_te, train_nb], axis=1)
print_step('Importing Data 12/19 7/7')
test_te = pd.concat([test_te, test_nb], axis=1)

print('~~~~~~~~~~~~~~~')
print_step('Run TE LGB')
params['num_rounds'] = 5200 #8100
print(train_te.shape)
print(test_te.shape)
results = run_cv_model(train_te, test_te, target, runLGB, params, rmse, 'te_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('te_lgb', pd.DataFrame({'te_lgb': results['train']}),
                        pd.DataFrame({'te_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_te_lgb.csv', index=False)
del train_te, test_te
gc.collect()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Importing Data 13/19')
drops = [x + '_' + str(y) for x in ['mobile', 'inception', 'nasnet'] for y in range(1, 11)] + \
        ['img_size_ratio', 'times_put_up_min', 'times_put_up_max'] + \
        ['embed' + str(i) for i in range(EMBED_SIZE)]
train_fe.drop(drops, axis=1, inplace=True)
test_fe.drop(drops, axis=1, inplace=True)

print_step('Embedding SVD 1/4')
svd = TruncatedSVD(n_components=NCOMP, algorithm='arpack')
svd.fit(train_embeddings_df)
print_step('Embedding SVD 2/4')
train_svd = pd.DataFrame(svd.transform(train_embeddings_df))
print_step('Embedding SVD 3/4')
test_svd = pd.DataFrame(svd.transform(test_embeddings_df))
print_step('Embedding SVD 4/4')
train_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
test_svd.columns = ['svd_embed_'+str(i+1) for i in range(NCOMP)]
train_fe = pd.concat([train_fe, train_svd], axis=1)
test_fe = pd.concat([test_fe, test_svd], axis=1)

print_step('Importing Data 6/19 1/3')
train_inter, test_inter = load_cache('interact')
print_step('Importing Data 6/19 2/3')
train_fe = pd.concat([train_fe, train_inter], axis=1)
print_step('Importing Data 6/19 3/3')
test_fe = pd.concat([test_fe, test_inter], axis=1)
train_fe.drop('item_id', axis=1, inplace=True)
test_fe.drop('item_id', axis=1, inplace=True)

print_step('Importing Data 14/19 1/2')
train_ridge, test_ridge = load_cache('tfidf_ridges')
print_step('Importing Data 14/19 2/2')
cols = [c for c in train_ridge.columns if 'ridge' in c]
train_fe[cols] = train_ridge[cols]
test_fe[cols] = test_ridge[cols]

print_step('Importing Data 15/19 1/3')
train_full_text_ridge, test_full_text_ridge = load_cache('full_text_ridge')
print_step('Importing Data 15/19 2/3')
train_fe['full_text_ridge'] = train_full_text_ridge['full_text_ridge']
print_step('Importing Data 15/19 3/3')
test_fe['full_text_ridge'] = test_full_text_ridge['full_text_ridge']

print_step('Importing Data 16/19 1/3')
train_complete_ridge, test_complete_ridge = load_cache('complete_ridge')
print_step('Importing Data 16/19 2/3')
train_fe['complete_ridge'] = train_complete_ridge['complete_ridge']
print_step('Importing Data 16/19 3/3')
test_fe['complete_ridge'] = test_complete_ridge['complete_ridge']

print_step('Importing Data 17/19 1/4')
train_pcat_ridge, test_pcat_ridge = load_cache('parent_cat_ridges')
print_step('Importing Data 17/19 2/4')
train_pcat_ridge = train_pcat_ridge[[c for c in train_pcat_ridge.columns if 'ridge' in c]]
test_pcat_ridge = test_pcat_ridge[[c for c in test_pcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 17/19 3/4')
train_fe = pd.concat([train_fe, train_pcat_ridge], axis=1)
print_step('Importing Data 17/19 4/4')
test_fe = pd.concat([test_fe, test_pcat_ridge], axis=1)

print_step('Importing Data 18/19 1/4')
train_rcat_ridge, test_rcat_ridge = load_cache('parent_regioncat_ridges')
print_step('Importing Data 18/19 2/4')
train_rcat_ridge = train_rcat_ridge[[c for c in train_rcat_ridge.columns if 'ridge' in c]]
test_rcat_ridge = test_rcat_ridge[[c for c in test_rcat_ridge.columns if 'ridge' in c]]
print_step('Importing Data 18/19 3/4')
train_fe = pd.concat([train_fe, train_rcat_ridge], axis=1)
print_step('Importing Data 18/19 4/4')
test_fe = pd.concat([test_fe, test_rcat_ridge], axis=1)

print_step('Importing Data 19/19 1/4')
train_catb_ridge, test_catb_ridge = load_cache('cat_bin_ridges')
print_step('Importing Data 19/19 2/4')
train_catb_ridge = train_catb_ridge[[c for c in train_catb_ridge.columns if 'ridge' in c]]
test_catb_ridge = test_catb_ridge[[c for c in test_catb_ridge.columns if 'ridge' in c]]
print_step('Importing Data 19/19 3/4')
train_fe = pd.concat([train_fe, train_catb_ridge], axis=1)
print_step('Importing Data 19/19 4/4')
test_fe = pd.concat([test_fe, test_catb_ridge], axis=1)

print('~~~~~~~~~~~~~~~~~~')
print_step('Run Ridge LGB')
print(train_fe.shape)
print(test_fe.shape)
params['num_rounds'] = 4900 #3100
results = run_cv_model(train_fe, test_fe, target, runLGB, params, rmse, 'ridge_lgb')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('ridge_lgb', pd.DataFrame({'ridge_lgb': results['train']}),
                           pd.DataFrame({'ridge_lgb': results['test']}))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print_step('Prepping submission file')
submission = pd.DataFrame()
submission['item_id'] = test_id
submission['deal_probability'] = results['test'].clip(0.0, 1.0)
submission.to_csv('submit/submit_ridge_lgb.csv', index=False)


# BASE
# [2018-06-21 15:24:00.453329] base_lgb cv scores : [0.21467065391115733, 0.21403075064933444, 0.21399333430251413, 0.2139006508898873, 0.21436928569495026]
# [2018-06-21 15:24:00.453394] base_lgb mean cv score : 0.2141929350895687
# [2018-06-21 15:24:00.453490] base_lgb std cv score : 0.0002866667503387072

# [100]   training's rmse: 0.22229        valid_1's rmse: 0.225697
# [200]   training's rmse: 0.215696       valid_1's rmse: 0.222124
# [300]   training's rmse: 0.211445       valid_1's rmse: 0.220422
# [400]   training's rmse: 0.207927       valid_1's rmse: 0.219243
# [500]   training's rmse: 0.204969       valid_1's rmse: 0.218502
# [600]   training's rmse: 0.202296       valid_1's rmse: 0.217956
# [700]   training's rmse: 0.199856       valid_1's rmse: 0.217497
# [800]   training's rmse: 0.197617       valid_1's rmse: 0.217151
# [900]   training's rmse: 0.195467       valid_1's rmse: 0.216867
# [1000]  training's rmse: 0.193399       valid_1's rmse: 0.216609
# [1100]  training's rmse: 0.191432       valid_1's rmse: 0.216389
# [1200]  training's rmse: 0.189532       valid_1's rmse: 0.216199
# [1300]  training's rmse: 0.187702       valid_1's rmse: 0.216057
# [1400]  training's rmse: 0.185982       valid_1's rmse: 0.215892
# [1500]  training's rmse: 0.184346       valid_1's rmse: 0.215758
# [1600]  training's rmse: 0.182705       valid_1's rmse: 0.215658
# [1700]  training's rmse: 0.181253       valid_1's rmse: 0.215575
# [1800]  training's rmse: 0.179763       valid_1's rmse: 0.215489
# [1900]  training's rmse: 0.178362       valid_1's rmse: 0.215417
# [2000]  training's rmse: 0.176916       valid_1's rmse: 0.215329
# [2100]  training's rmse: 0.175516       valid_1's rmse: 0.215266
# [2200]  training's rmse: 0.174251       valid_1's rmse: 0.215213
# [2300]  training's rmse: 0.172937       valid_1's rmse: 0.215157
# [2400]  training's rmse: 0.171678       valid_1's rmse: 0.215108
# [2500]  training's rmse: 0.170459       valid_1's rmse: 0.215058
# [2600]  training's rmse: 0.169241       valid_1's rmse: 0.215015
# [2700]  training's rmse: 0.168042       valid_1's rmse: 0.214979
# [2800]  training's rmse: 0.16692        valid_1's rmse: 0.214944
# [2900]  training's rmse: 0.165788       valid_1's rmse: 0.214909
# [3000]  training's rmse: 0.164674       valid_1's rmse: 0.214882
# [3100]  training's rmse: 0.163611       valid_1's rmse: 0.21485
# [3200]  training's rmse: 0.162535       valid_1's rmse: 0.214834
# [3300]  training's rmse: 0.161554       valid_1's rmse: 0.214812
# [3400]  training's rmse: 0.160565       valid_1's rmse: 0.214786
# [3500]  training's rmse: 0.159592       valid_1's rmse: 0.21477
# [3600]  training's rmse: 0.158631       valid_1's rmse: 0.21475
# [3700]  training's rmse: 0.157667       valid_1's rmse: 0.214731
# [3800]  training's rmse: 0.156726       valid_1's rmse: 0.214712
# [3900]  training's rmse: 0.155777       valid_1's rmse: 0.214702
# [4000]  training's rmse: 0.154786       valid_1's rmse: 0.214688
# [4100]  training's rmse: 0.153876       valid_1's rmse: 0.214678
# [4200]  training's rmse: 0.152994       valid_1's rmse: 0.214671


# WITH TE
# [2018-06-21 01:55:12.384642] te_lgb cv scores : [0.21408321765643631, 0.21349100783866126, 0.2134437589434402, 0.2132399990935228, 0.21394712169608773]
# [2018-06-21 01:55:12.384708] te_lgb mean cv score : 0.21364102104562965
# [2018-06-21 01:55:12.384805] te_lgb std cv score : 0.000319834443459895

# [100]   training's rmse: 0.222398       valid_1's rmse: 0.225094
# [200]   training's rmse: 0.216948       valid_1's rmse: 0.221632
# [300]   training's rmse: 0.213441       valid_1's rmse: 0.219966
# [400]   training's rmse: 0.21062        valid_1's rmse: 0.218914
# [500]   training's rmse: 0.208119       valid_1's rmse: 0.218143
# [600]   training's rmse: 0.205885       valid_1's rmse: 0.217576
# [700]   training's rmse: 0.203809       valid_1's rmse: 0.217115
# [800]   training's rmse: 0.201833       valid_1's rmse: 0.21674
# [900]   training's rmse: 0.199981       valid_1's rmse: 0.216444
# [1000]  training's rmse: 0.198191       valid_1's rmse: 0.216188
# [1100]  training's rmse: 0.196513       valid_1's rmse: 0.215974
# [1200]  training's rmse: 0.194878       valid_1's rmse: 0.215776
# [1300]  training's rmse: 0.193308       valid_1's rmse: 0.215601
# [1400]  training's rmse: 0.191793       valid_1's rmse: 0.215459
# [1500]  training's rmse: 0.190314       valid_1's rmse: 0.215325
# [1600]  training's rmse: 0.188911       valid_1's rmse: 0.21522
# [1700]  training's rmse: 0.187529       valid_1's rmse: 0.21511
# [1800]  training's rmse: 0.186213       valid_1's rmse: 0.215013
# [1900]  training's rmse: 0.184928       valid_1's rmse: 0.214928
# [2000]  training's rmse: 0.183655       valid_1's rmse: 0.214847
# [2100]  training's rmse: 0.182423       valid_1's rmse: 0.214777
# [2200]  training's rmse: 0.181233       valid_1's rmse: 0.214729
# [2300]  training's rmse: 0.180066       valid_1's rmse: 0.214666
# [2400]  training's rmse: 0.178933       valid_1's rmse: 0.214618
# [2500]  training's rmse: 0.17782        valid_1's rmse: 0.214571
# [2600]  training's rmse: 0.176744       valid_1's rmse: 0.214532
# [2700]  training's rmse: 0.175679       valid_1's rmse: 0.214489
# [2800]  training's rmse: 0.174638       valid_1's rmse: 0.214447
# [2900]  training's rmse: 0.173602       valid_1's rmse: 0.214411
# [3000]  training's rmse: 0.17259        valid_1's rmse: 0.214367
# [3100]  training's rmse: 0.171606       valid_1's rmse: 0.214333
# [3200]  training's rmse: 0.170596       valid_1's rmse: 0.214298
# [3300]  training's rmse: 0.169647       valid_1's rmse: 0.214288
# [3400]  training's rmse: 0.168714       valid_1's rmse: 0.214264
# [3500]  training's rmse: 0.167782       valid_1's rmse: 0.214235
# [3600]  training's rmse: 0.166849       valid_1's rmse: 0.214204
# [3700]  training's rmse: 0.16591        valid_1's rmse: 0.214189
# [3800]  training's rmse: 0.164998       valid_1's rmse: 0.214161
# [3900]  training's rmse: 0.164102       valid_1's rmse: 0.214141
# [4000]  training's rmse: 0.163213       valid_1's rmse: 0.214114
# [4100]  training's rmse: 0.162342       valid_1's rmse: 0.214097
# [4200]  training's rmse: 0.161493       valid_1's rmse: 0.214083


# WITH RIDGES
# [2018-06-21 11:03:11.528913] ridge_lgb cv scores : [0.213991140517113, 0.21315434514289164, 0.21309603201930144, 0.21303281451462452, 0.21346750118045446]
# [2018-06-21 11:03:11.528980] ridge_lgb mean cv score : 0.21334836667487705
# [2018-06-21 11:03:11.529080] ridge_lgb std cv score : 0.0003544690513431004
