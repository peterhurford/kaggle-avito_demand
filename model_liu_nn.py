import gc
import re
import string

import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Dense, PReLU, Flatten, Layer
from keras.layers import GaussianDropout, BatchNormalization
from keras.layers import Input, Dropout, Lambda, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GRU
from keras.layers import SpatialDropout1D,GlobalAveragePooling1D
from keras.layers import Convolution1D, GlobalMaxPooling1D, Conv2D, GlobalMaxPooling2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import text, sequence
from sklearn.model_selection import KFold
import time
from gensim.models.keyedvectors import KeyedVectors
import env_check
from attent import Attention

notebookstart = time.time()

debug = 0
USE_KFLOD = 1
USE_IMAGE = 1
max_features = 100000
title_maxlen = 50
desp_maxlen = 100
embed_size = 200



print("Data Load Stage")
training = pd.read_pickle('../input/train.pickle')
testing = pd.read_pickle('../input/test.pickle')
training['auto_incre'] = range(training.shape[0])
testing['auto_incre'] = range(testing.shape[0])
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
del train_img_fea, test_img_fea, fea_names

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

del fea_names, train_img_score, test_img_score
gc.collect()

if debug:
  training = training.sample(frac=0.01)
  testing = testing.sample(frac=0.01)

traindex = training.index
testdex = testing.index

training = training.merge(gp, on='user_id', how='left')
testing = testing.merge(gp, on='user_id', how='left')
del gp
gc.collect()
training.index = traindex
testing.index = testdex

for col in ['avg_days_up_user', 'avg_times_up_user', 'n_user_items']:
  mmean = np.nanmean(training[col].values)
  training[col].fillna(mmean, inplace=True)
  testing[col].fillna(mmean, inplace=True)

y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = training.append(testing)
df_index = df.index
del training, testing
gc.collect()
print('All Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
price_mean = np.nanmean(df["price"].values)
df["price"].fillna(price_mean, inplace=True)
df['price'] = np.log(df["price"] + 0.001)
df['item_seq_number'] = np.log(df["item_seq_number"] + 0.001)

df["image_top_1"].fillna(-999, inplace=True)
df['no_img'] = df['image'].isnull().astype('uint8')
df['no_desp'] = df['description'].isnull().astype('uint8')
df['no_p1'] = df['param_1'].isnull().astype('uint8')
df['no_p2'] = df['param_2'].isnull().astype('uint8')
df['no_p3'] = df['param_3'].isnull().astype('uint8')
print("Create Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday

print("Create Total Param Feature")
df['param_1'] = df['param_1'].fillna('Другое')
df['param_2'] = df['param_2'].fillna('Другое')
df['param_3'] = df['param_3'].fillna('Другое')
df['text_feat'] = df['param_1'].astype(str) + df['param_2'].astype(str) + \
                  df['param_3'].astype(str)

print("Create Meta Text Feature")
textfeats = ["description", "text_feat", "title"]
count_set = lambda l1, l2: sum([1 for x in l1 if x in l2])
russian_stop = set(stopwords.words('russian'))
for cols in textfeats:
  df[cols] = df[cols].astype(str).fillna('nicapotato')
  df[cols + '_cap_num'] = df[cols].apply(
    lambda comment: len(re.findall(r'[A-ZА-Я]', comment)))
  df[cols + '_cap_num'] = np.log(df[cols + '_cap_num'] + 0.001)
  df[cols] = df[cols].str.lower()  # FILL NA
  df[cols + '_len'] = df[cols].apply(len)  # Count number of Characters
  df[cols + '_len'] = np.log(df[cols + '_len'] + 0.001)
  df[cols + '_num_words'] = df[cols].apply(
    lambda comment: len(comment.split()))  # Count number of Words
  df[cols + '_num_words'] = np.log(df[cols + '_num_words'] + 0.001)
  df[cols + '_punc_count'] = df[cols].apply(
    lambda com: count_set(com, set(string.punctuation)))
  df[cols + '_punc_count'] = np.log(df[cols + '_punc_count'] + 0.001)
  df[cols + '_digit_count'] = df[cols].apply(
    lambda com: count_set(com, set(string.digits)))
  df[cols + '_digit_count'] = np.log(df[cols + '_digit_count'] + 0.001)
  df[cols + '_num_unique_words'] = df[cols].apply(
    lambda comment: len(set(w for w in comment.split())))
  df[cols + '_num_unique_words'] = np.log(
    df[cols + '_num_unique_words'] + 0.001)
  df[cols + '_stop_word_count'] = df[cols].apply(
    lambda x: len([w for w in str(x).lower().split() if w in russian_stop]))
  df[cols + '_stop_word_count'] = np.log(
    df[cols + '_stop_word_count'] + 0.001)

# Create Validation Index and Remove Dead Variables
training_index = df.loc[
  df.activation_date <= pd.to_datetime('2017-04-07')].index
validation_index = df.loc[
  df.activation_date >= pd.to_datetime('2017-04-08')].index
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("Encode Variables")
categorical = ["param_1", "param_2", "param_3", "region", "city",
               "parent_category_name",
               "category_name", "user_type", "image_top_1"]
print(f"Encoding : {categorical}")
categorical_value_count = []
# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
  df[col] = lbl.fit_transform(df[col].astype(str))
  categorical_value_count.append(len(lbl.classes_))
df['user_id'] = lbl.fit_transform(df['user_id'].astype(str))

origin_index = df.index.values

# Count based feature
gp = df[['city', 'user_id']].groupby(
  by=['city'])[['user_id']].count().reset_index() \
  .rename(index=str, columns={'user_id': 'city_count'})
df = df.merge(gp, on=['city'], how='left')

gp = df[['region', 'user_id']].groupby(
  by=['region'])[['user_id']].count().reset_index() \
  .rename(index=str, columns={'user_id': 'region_count'})
df = df.merge(gp, on=['region'], how='left')

gp = df[['user_id', 'city']].groupby(
  by=['user_id'])[['city']].count().reset_index() \
  .rename(index=str, columns={'city': 'user_ad_count'})
df = df.merge(gp, on=['user_id'], how='left')
df['user_ad_count'] = np.log(df['user_ad_count'] + 0.001)

del df['user_id']

def parse(x):
  x['level'] = pd.qcut(x['price'], 100, labels=False, duplicates='drop')
  return x['level']

print("Generate Level Feature")
gp = df[['category_name', 'price']].groupby(
  by=['category_name']).apply(parse).reset_index()
df = df.merge(gp[['level']], how='left', left_index=True, right_index=True)
df["level"].fillna(np.nanmean(df["level"].values), inplace=True)
df['level'] = np.log(df['level'] + 0.001)

df['category_level'] = df['category_name'].astype('str') \
                       + "_" + df['level'].astype(str)
df['category_level_encode'] = lbl.fit_transform(df['category_level'])

# del df['level']
del df['category_level']
df.index = df_index

features = [x for x in df.columns.values if x not in ['title', 'description', 'text_feat', 'auto_incre']]
print('ok')
train_df = df.loc[traindex, :]
test_df = df.loc[testdex, :]

binary = ['no_img', 'no_desp', 'no_p1', 'no_p2', 'no_p3',]

categorical = categorical
continous = [col for col in features if col not in (categorical + binary)]
scaler = StandardScaler()
df[continous] = scaler.fit_transform(df[continous])
continous = continous + binary



print("text embedding")
tokenizer = text.Tokenizer(num_words=max_features)

print('fitting tokenizer')
train_df['description'] = train_df['description'].astype(str)
tokenizer.fit_on_texts(list(train_df['description'].fillna('NA').values) +
                       list(train_df['title'].fillna('NA').values))


def get_coefs(w, *arr):
    return w, np.asarray(arr, dtype='float32')


print('getting embeddings')

word2vec = KeyedVectors.load_word2vec_format('../input/word2vec.bin', binary=True, unicode_errors='ignore')

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)+1)
embedding_matrix = np.zeros((nb_words, embed_size))
oof, found = 0, 0
for word, i in word_index.items():
  if i >= max_features: continue
  if word in word2vec.wv:
    found += 1
    embedding_matrix[i] = word2vec.wv[word]
  else:
    oof += 1
print(f"Get embedding finish. {found} words FOUND. {oof} words NOT FOUND.")
del word2vec
gc.collect()


def rmse(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))

class HeadAverage(Layer):
  def __init__(self, average_range):
    super(HeadAverage, self).__init__()
    self.average_range = average_range

  def call(self, x, **kwargs):
    return K.mean(x[:,:self.average_range,:], axis=1)

  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[2]


class AvitoModel:
  def __init__(self):
    super(AvitoModel, self).__init__()

    self.categorical_num = categorical_value_count

  def build_model(self):
    categorial_inp = Input(shape=(len(categorical),))
    cat_embeds = []
    for idx, col in enumerate(categorical):
      x = Lambda(lambda x: x[:, idx, None])(categorial_inp)
      x = Embedding(categorical_value_count[idx], 30, input_length=1)(x)
      cat_embeds.append(x)
    embeds = concatenate(cat_embeds, axis=2)
    embeds = GaussianDropout(0.2)(embeds)
    embeds = Flatten()(embeds)
    continous_inp = Input(shape=(len(continous),))
    continous_out = BatchNormalization()(continous_inp)
    continous_out = Dense(100)(continous_out)
    continous_out = PReLU()(continous_out)

    def bi_gru(inp, embed_layer):
      rnn_size = 64
      drop_rate = 0.2
      GRU_CLASS = CuDNNGRU if env_check.is_linux() else GRU
      emb = embed_layer(inp)
      out = SpatialDropout1D(drop_rate)(emb)
      out = Bidirectional(GRU_CLASS(rnn_size, return_sequences=True))(out)
      out = Bidirectional(GRU_CLASS(rnn_size, return_sequences=True))(out)
      attent = Attention()(out)
      ave_pool = GlobalAveragePooling1D()(out)
      max_pool = GlobalMaxPooling1D()(out)
      out = concatenate([ave_pool, max_pool, attent], axis=1)
      return Dropout(drop_rate)(out)

    def cnn_model(inp, embed_layer):
      drop_rate = 0.2
      emb = embed_layer(inp)
      out = SpatialDropout1D(drop_rate)(emb)
      fileter_count = 100
      cnn1 = Convolution1D(fileter_count, 3, padding='same', strides=1,
                           activation='relu')(out)
      cnn1_max = GlobalMaxPooling1D()(cnn1)
      cnn1_ave = GlobalAveragePooling1D()(cnn1)
      cnn2 = Convolution1D(fileter_count, 4, padding='same', strides=1,
                           activation='relu')(out)
      cnn2_max = GlobalMaxPooling1D()(cnn2)
      cnn2_ave = GlobalAveragePooling1D()(cnn2)
      cnn3 = Convolution1D(fileter_count, 5, padding='same', strides=1,
                           activation='relu')(out)
      cnn3_max = GlobalMaxPooling1D()(cnn3)
      cnn3_ave = GlobalAveragePooling1D()(cnn3)

      cnn = concatenate(
        [cnn1_max, cnn1_ave, cnn2_max, cnn2_ave, cnn3_max, cnn3_ave], axis=-1)
      cnn = Dropout(drop_rate)(cnn)
      return cnn

    embed_layer = Embedding(nb_words, embed_size, weights=[embedding_matrix],
                      trainable=False)

    title_inp = Input(shape=(title_maxlen,))
    desp_inp = Input(shape=(desp_maxlen,))
    title = cnn_model(title_inp, embed_layer)
    desp = cnn_model(desp_inp, embed_layer)
    title_rnn = bi_gru(title_inp, embed_layer)
    desp_rnn = bi_gru(desp_inp, embed_layer)
    text_fea = concatenate([title, desp, title_rnn, desp_rnn], axis=1)
    text_fea = BatchNormalization()(text_fea)
    text_fea = Dense(200)(text_fea)
    text_fea = PReLU()(text_fea)
    text_fea = Dropout(0.05)(text_fea)

    concat_list = [embeds, continous_out, text_fea]
    input_list = [categorial_inp, continous_inp, title_inp, desp_inp]
    if USE_IMAGE:
      # Image input
      img_input = Input(shape=(64, 64, 3))
      img_cnn = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(img_input)
      img_cnn = BatchNormalization()(img_cnn)
      img_cnn = MaxPool2D()(img_cnn)
      img_cnn = Dropout(0.3)(img_cnn)
      img_cnn = Conv2D(filters=50, kernel_size=(3, 3), activation='relu')(img_cnn)
      img_cnn = BatchNormalization()(img_cnn)
      img_cnn = MaxPool2D()(img_cnn)
      img_cnn = Dropout(0.3)(img_cnn)
      img_cnn = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(img_cnn)
      img_cnn = BatchNormalization()(img_cnn)
      img_cnn = MaxPool2D()(img_cnn)
      img_cnn = Dropout(0.2)(img_cnn)
      img_output = Flatten()(img_cnn)
      img_output = Dense(200)(img_output)

      input_list.append(img_input)
      concat_list.append(img_output)

    x = concatenate(concat_list, axis=1)
    x = BatchNormalization()(x)
    x = Dense(200)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dense(50)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    outp = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_list,
                  output=outp)

    model.compile(loss=rmse,
                  optimizer=Adam(lr=0.001),
                  metrics=[rmse])
    print(model.summary())
    return model


def get_input(df, train=True):
  title = tokenizer.texts_to_sequences(df['title'])
  title = sequence.pad_sequences(title, maxlen=title_maxlen)
  desp = tokenizer.texts_to_sequences(df['description'])
  desp = sequence.pad_sequences(desp, maxlen=desp_maxlen)
  get_input_list = [df[categorical], df[continous], title, desp]
  if USE_IMAGE:
    if train:
      img = np.load('../input/train_img.npy', mmap_mode='r')
    else:
      img = np.load('../input/test_img.npy', mmap_mode='r')
    img = img[df['auto_incre']]
    get_input_list.append(img)
  return get_input_list

def save_prediction(pred, file):
  sub = pd.DataFrame(pred, columns=["deal_probability"], index=testdex)
  sub['deal_probability'] = sub['deal_probability'].clip(0.0, 1.0)
  sub.to_csv(env_check.log_path + f"/{file}.csv", index=True, header=True)


fold_count = 10
skf = KFold(fold_count, shuffle=True, random_state=12345)
gen = skf.split(train_df, y)
if not USE_KFLOD:
  fold_count = 1

test_prediction = np.zeros((fold_count, test_df.shape[0]))

last_clf = None
for index, (train_index, valid_index) in enumerate(gen):
  gc.collect()
  if type(train_df) is pd.DataFrame:
    X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
  else:
    X_train, X_valid = train_df[train_index], train_df[valid_index]
  y_train, y_valid = y[train_index], y[valid_index]
  print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>> {index} KFlod.....")
  print(f"X_train.shape {X_train.shape} X_valid.shape {X_valid.shape}")

  modelstart = time.time()
  model = AvitoModel().build_model()
  file_path = env_check.log_path + f"/keras_nn_{index}.hdf5"
  check_point = ModelCheckpoint(file_path, monitor='val_rmse', mode='min',
                                save_best_only=True, verbose=1)
  early_stop = EarlyStopping(monitor='val_rmse', patience=3, mode='min')
  rlrop = ReduceLROnPlateau(monitor='val_rmse',mode='auto',patience=1,verbose=1,factor=0.5,cooldown=0,min_lr=1e-6)
  model.fit(get_input(X_train), y_train, batch_size=256, epochs=8, validation_data=(get_input(X_valid), y_valid), verbose=2, callbacks=[check_point, early_stop, rlrop])
  gc.collect()
  model.load_weights(file_path)
  prediction = model.predict(get_input(test_df, train=False), batch_size=128)
  save_prediction(prediction, f'kfold_{index}')
  prediction = np.reshape(prediction, (-1))
  test_prediction[index] = prediction
  if index == fold_count - 1:
    break

  print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))


print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))

lgpred = test_prediction.mean(axis=0)

save_prediction(lgpred, 'lgsub')

