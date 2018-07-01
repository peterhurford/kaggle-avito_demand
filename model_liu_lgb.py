import gc
import os
import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Models Packages
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

from sklearn.metrics import mean_squared_error
from math import sqrt
import env_check
import re

notebookstart = time.time()

debug = 0
USE_TFIDF = 1
USE_KFLOD = 1
SEED = 42


class SklearnWrapper(object):
  def __init__(self, clf, seed=0, params=None, seed_bool=True):
    if (seed_bool == True):
      params['random_state'] = seed
    self.clf = clf(**params)

  def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)

  def predict(self, x):
    return self.clf.predict(x)


def get_oof(clf, x_train, y, x_test):
  NFOLDS = 5
  kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
  oof_train = np.zeros((ntrain,))
  oof_test = np.zeros((ntest,))
  oof_test_skf = np.empty((NFOLDS, ntest))

  for i, (train_index, test_index) in enumerate(kf.split(range(x_train.shape[0]))):
    print('\nFold {}'.format(i))
    x_tr = x_train[train_index]
    y_tr = y[train_index]
    x_te = x_train[test_index]

    clf.train(x_tr, y_tr)

    oof_train[test_index] = clf.predict(x_te)
    oof_test_skf[i, :] = clf.predict(x_test)

  oof_test[:] = oof_test_skf.mean(axis=0)
  return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def cleanName(text):
  try:
    textProc = text.lower()
    textProc = " ".join(map(str.strip, re.split('(\d+)', textProc)))
    regex = re.compile(u'[^[:alpha:]]')
    textProc = regex.sub(" ", textProc)
    textProc = " ".join(textProc.split())
    return textProc
  except:
    return "name error"



def print_importance(bst):
  names = bst.feature_name()
  imps = bst.feature_importance()
  for n, i in sorted(zip(names, imps), key=lambda x: -x[1]):
    if i == 0:
      break
    print(f'{n} {i}')


def rmse(y, y0):
  assert len(y) == len(y0)
  return np.sqrt(np.mean(np.power((y - y0), 2)))


def add_noise(series, noise_level):
  return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0.0):
  """
  Smoothing is computed like in the following paper by Daniele Micci-Barreca
  https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
  trn_series : training categorical feature as a pd.Series
  tst_series : test categorical feature as a pd.Series
  target : target data as a pd.Series
  min_samples_leaf (int) : minimum samples to take category average into account
  smoothing (int) : smoothing effect to balance categorical average vs prior
  """
  assert len(trn_series) == len(target)
  assert trn_series.name == tst_series.name
  temp = pd.concat([trn_series, target], axis=1)
  # Compute target mean
  averages = temp.groupby(by=trn_series.name)[target.name].agg(
    ["mean", "count"])
  # Compute smoothing
  smoothing = 1 / (
        1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
  # Apply average function to all target data
  prior = target.mean()
  # The bigger the count the less full_avg is taken into account
  averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
  averages.drop(["mean", "count"], axis=1, inplace=True)
  # Apply averages to trn and tst series
  ft_trn_series = pd.merge(
    trn_series.to_frame(trn_series.name),
    averages.reset_index().rename(
      columns={'index': target.name, target.name: 'average'}),
    on=trn_series.name,
    how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
  # pd.merge does not keep the index so restore it
  ft_trn_series.index = trn_series.index
  ft_tst_series = pd.merge(
    tst_series.to_frame(tst_series.name),
    averages.reset_index().rename(
      columns={'index': target.name, target.name: 'average'}),
    on=tst_series.name,
    how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
  # pd.merge does not keep the index so restore it
  ft_tst_series.index = tst_series.index
  return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series,
                                                          noise_level)



print("\nData Load Stage")
training = pd.read_pickle('../input/train.pickle')
testing = pd.read_pickle('../input/test.pickle')
gp = pd.read_csv('../input/aggregated_features.csv')
train_img_fea = np.load(f'../input/train_image_meta_wihtout_zero_size.npy')
test_img_fea = np.load(f'../input/test_image_meta.npy')

fea_names = []
for index in range(train_img_fea.shape[1]):
  fea_name = f'img_fea_{index}'
  fea_names.append(fea_name)
  training[fea_name] = 0
  testing[fea_name] = 0

training.loc[training['image'].notnull(), fea_names] = train_img_fea
testing.loc[testing['image'].notnull(), fea_names] = test_img_fea

train_img_score = np.load(f'../input/train_img_score.npy')
test_img_score = np.load(f'../input/test_img_score.npy')

fea_names = []
for index in range(train_img_score.shape[1]):
  fea_name = f'img_score_{index}'
  fea_names.append(fea_name)
  training[fea_name] = 0
  testing[fea_name] = 0


training.loc[training['image'].notnull(), fea_names] = train_img_score
testing.loc[testing['image'].notnull(), fea_names] = test_img_score

del train_img_fea, test_img_fea, fea_names
gc.collect()

if debug:
  training = training.sample(frac=0.01, random_state=12345)
  testing = testing.sample(frac=0.01, random_state=12345)

traindex = training.index
testdex = testing.index

training = training.merge(gp, on='user_id', how='left')
testing = testing.merge(gp, on='user_id', how='left')
del gp
gc.collect()

categorical = ["region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1", "param_1",
               "param_2", "param_3"]

training.index = traindex
testing.index = testdex

for col in ['avg_days_up_user', 'avg_times_up_user', 'n_user_items']:
  mmean = np.nanmean(training[col].values)
  training[col].fillna(mmean, inplace=True)
  testing[col].fillna(mmean, inplace=True)

ntrain = training.shape[0]
ntest = testing.shape[0]


training.index = traindex
testing.index = testdex

y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

for col in categorical:
  trn, sub = target_encode(training[col],
                           testing[col],
                           target=y,
                           min_samples_leaf=100,
                           smoothing=10,
                           noise_level=0.01)
  training[f'{col}_encoding'] = trn
  testing[f'{col}_encoding'] = sub


print("Combine Train and Test")
df = training.append(testing)
df_index = df.index
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df['no_image'] = df['image_top_1'].isnull().astype('uint8')
df['no_price'] = df['price'].isnull().astype('uint8')
df["image_top_1"].fillna(-999, inplace=True)


def get_wo_nan_price(df):
  df_wo_nan = pd.DataFrame(index=df.index)
  df_wo_nan['price_imputed'] = df.groupby(
    ['region', 'city', 'parent_category_name', 'category_name', 'param_1',
     'param_2', 'param_3'])['price'].apply(lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = df.groupby(
    ['region', 'city', 'parent_category_name', 'category_name', 'param_1',
     'param_2'])['price'].apply(lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = df.groupby(
    ['region', 'city', 'parent_category_name', 'category_name', 'param_1'])[
    'price'].apply(lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = \
  df.groupby(['region', 'city', 'parent_category_name', 'category_name'])[
    'price'].apply(lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = \
  df.groupby(['region', 'city', 'parent_category_name'])['price'].apply(
    lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = df.groupby(['region', 'city'])['price'].apply(
    lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  df_wo_nan['price_imputed'] = df.groupby(['region'])['price'].apply(
    lambda x: x.fillna(x.median()))
  print(df_wo_nan['price_imputed'].isnull().sum())
  return df_wo_nan


price_imputed = get_wo_nan_price(df)
df = df.merge(price_imputed, left_index=True, right_index=True, how='left')

df["price"] = df["price_imputed"]
del df["price_imputed"]
df["price"] = np.log(df["price"] + 0.001)


print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday

# Create Validation Index and Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nText Features")

# Feature Engineering

# Meta Text Features
textfeats = ["description", "title"]

# df['title'] = df['title'].apply(lambda x: cleanName(x))
# df["description"]   = df["description"].apply(lambda x: cleanName(x))

for cols in textfeats:
  df[cols] = df[cols].astype(str)
  df[cols] = df[cols].astype(str).fillna('missing')  # FILL NA
  df[cols] = df[
    cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
  df[cols + '_num_words'] = df[cols].apply(
    lambda comment: len(comment.split()))  # Count number of Words
  df[cols + '_num_unique_words'] = df[cols].apply(
    lambda comment: len(set(w for w in comment.split())))
  df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[
    cols + '_num_words'] * 100  # Count Unique Words

# Count based feature
gp = df[['city', 'user_id']].groupby(
  by=['city'])[['user_id']].count().reset_index() \
  .rename(index=str, columns={'user_id': 'city_count'})
df = df.merge(gp, on=['city'], how='left')

gp = df[['region', 'user_id']].groupby(
  by=['region'])[['user_id']].count().reset_index() \
  .rename(index=str, columns={'user_id': 'region_count'})
df = df.merge(gp, on=['region'], how='left')

gp = df[['param_1', 'user_id']].groupby(
  by=['param_1'])[['user_id']].count().reset_index() \
  .rename(index=str, columns={'user_id': 'param_1_count'})
df = df.merge(gp, on=['param_1'], how='left')


gp = df[['user_id', 'city']].groupby(
  by=['user_id'])[['city']].count().reset_index() \
  .rename(index=str, columns={'city': 'user_ad_count'})
df = df.merge(gp, on=['user_id'], how='left')
df['user_ad_count'] = np.log(df['user_ad_count'] + 0.001)

gp = df[['city', 'price']].groupby(by=['city'])[['price']].mean().reset_index()\
  .rename(index=str, columns={'price': "city_mean_price"})
df = df.merge(gp, on=['city'], how='left')

gp = df[['region', 'price']].groupby(by=['region'])[['price']].mean().reset_index()\
  .rename(index=str, columns={'price': "region_mean_price"})
df = df.merge(gp, on=['region'], how='left')

gp = df[['user_id', 'price']].groupby(by=['user_id'])[['price']].mean().reset_index()\
  .rename(index=str, columns={'price': "user_id_mean_price"})
df = df.merge(gp, on=['user_id'], how='left')

del df['user_id']

def parse(x):
  x['level'] = pd.qcut(x['price'], 100, labels=False, duplicates='drop')
  return x['level']

print("Generate Level Feature")

lbl = preprocessing.LabelEncoder()
gp = df[['category_name', 'price']].groupby(
  by=['category_name']).apply(parse).reset_index()
df = df.merge(gp[['level']], how='left', left_index=True, right_index=True)
df['level'] = np.log(df['level'] + 0.001)

df['category_level'] = df['category_name'].astype('str') \
                       + "_" + df['level'].astype(str)
df['category_level_encode'] = lbl.fit_transform(df['category_level'])

# del df['level']
del df['category_level']





print("\nEncode Variables")
print("Encoding :", categorical)

# Encoder:
max_bin = 200
categorical_copy = categorical.copy()
for col in categorical_copy:
  df[col].fillna('Unknown')
  col_val = df[col].astype(str)
  res = lbl.fit_transform(col_val)
  types = len(lbl.classes_)
  col_count = int(np.ceil(types/max_bin))
  for i in range(col_count):
    f_name = f'{col}_{i}'
    df[f_name] = np.where(res // max_bin == i, res - max_bin * i, 0)
    categorical.append(f_name)
  del df[col]
  categorical.remove(col)


df.index = df_index


print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
  "stop_words": russian_stop,
  "analyzer": 'word',
  "token_pattern": r'\w{1,}',
  "sublinear_tf": True,
  "dtype": np.float32,
  "norm": 'l2',
  # "min_df":5,
  # "max_df":.9,
  "smooth_idf": False
}


def get_col(col_name): return lambda x: x[col_name]


##I added to the max_features of the description. It did not change my score much but it may be worth investigating
title_vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                  max_features=7000,
                                  **tfidf_para,)
desp_vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                  max_features=17000,
                                  **tfidf_para,)

start_vect = time.time()
title_vect_df = title_vectorizer.fit_transform(df['title'])
desp_vect_df = desp_vectorizer.fit_transform(df['description'])

n_comp = 6
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
title_svd = svd_obj.fit_transform(title_vect_df)
print(svd_obj.explained_variance_ratio_)

for i in range(n_comp):
  df[f'svd_title_{i}'] = title_svd[:, i]
del title_svd

ready_df = hstack((title_vect_df, desp_vect_df), format='csr')
tfvocab = title_vectorizer.get_feature_names()
tfvocab.extend(desp_vectorizer.get_feature_names())
print(len(tfvocab))
print(
  "Vectorization Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1, inplace=True)


ridge_file_path = '../input/all_ridge_pred.npy'
if os.path.exists(ridge_file_path) and not debug:
  ridge_pred = np.load(ridge_file_path)
else:
  ridge_params = {'alpha': 20.0, 'fit_intercept': True, 'normalize': False,
                  'copy_X': True,
                  'max_iter': None, 'tol': 0.001, 'solver': 'auto',
                  'random_state': SEED}
  ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
  ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y,
                                            ready_df[ntrain:])

  rms = sqrt(mean_squared_error(y, ridge_oof_train))
  print('Ridge OOF RMSE: {}'.format(rms))
  print("Modeling Stage")

  ridge_pred = np.concatenate([ridge_oof_train, ridge_oof_test])

  if not debug:
    np.save(ridge_file_path, ridge_pred)

df['ridge_preds'] = ridge_pred

# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex, :].values),
            ready_df[0:traindex.shape[0]]]).tocsr()  # Sparse Matrix
testing = hstack(
  [csr_matrix(df.loc[testdex, :].values), ready_df[traindex.shape[0]:]]).tocsr()
tfvocab = df.columns.tolist() + tfvocab
for shape in [X, testing]:
  print("{} Rows and {} Cols".format(*shape.shape))
print(f"Feature Names Length: {len(tfvocab)} feature list: {df.columns.tolist()}")
del df
gc.collect()


def save_prediction(pred, file):
  sub = pd.DataFrame(pred, columns=["deal_probability"], index=testdex)
  sub['deal_probability'] = sub['deal_probability'].clip(0.0, 1.0)
  sub.to_csv(env_check.log_path + f"/{file}.csv", index=True, header=True)


fold_count = 5
skf = KFold(fold_count, shuffle=True, random_state=12345)
gen = skf.split(X, y)

if not USE_KFLOD:
  fold_count = 1

test_prediction = np.zeros((fold_count, testing.shape[0]))

last_clf = None
for index, (train_index, valid_index) in enumerate(gen):
  if type(X) is pd.DataFrame:
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
  else:
    X_train, X_valid = X[train_index], X[valid_index]
  y_train, y_valid = y[train_index], y[valid_index]
  print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>> {index} KFlod.....")
  print(f"X_train.shape {X_train.shape} X_valid.shape {X_valid.shape}")
  lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 500,
    'colsample_bytree': 0.7,
    'subsample': 0.5,
    # 'bagging_freq': 5,
    'learning_rate': 0.02,
    'verbose': 0,
  }
  if env_check.is_linux():
    lgbm_params.update({'device': 'gpu'})

  # LGBM Dataset Formatting
  lgtrain = lgb.Dataset(X_train, y_train,
                        feature_name=tfvocab,
                        categorical_feature=categorical)
  lgvalid = lgb.Dataset(X_valid, y_valid,
                        feature_name=tfvocab,
                        categorical_feature=categorical)

  # Go Go Go
  modelstart = time.time()
  lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=160000,
    valid_sets=[lgvalid],
    valid_names=['valid'],
    early_stopping_rounds=50,
    verbose_eval=50
  )
  lgb_clf.save_model(env_check.log_path + f'/lgb_{index}.bin')
  test_prediction[index] = lgb_clf.predict(testing)
  save_prediction(test_prediction[index], f'lbg_fold_{index}')
  if index == fold_count - 1:
    last_clf = lgb_clf
    break

  print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))


print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))

lgpred = test_prediction.mean(axis=0)

lgsub = pd.DataFrame(lgpred, columns=["deal_probability"], index=testdex)
lgsub['deal_probability'] = lgsub['deal_probability'].clip(0.0, 1.0)
lgsub.to_csv(env_check.log_path + "/lgsub.csv", index=True, header=True)
print_importance(last_clf)




