import string

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import print_step, Scaler
from cache import get_data, is_in_cache, load_cache, save_in_cache


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

if not is_in_cache('data_with_fe'):
    print('~~~~~~~~~~~~')
    print_step('Merging')
    merge = pd.concat([train, test])

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Imputation 1/7')
    merge['param_1'].fillna('missing', inplace=True)
    print_step('Imputation 2/7')
    merge['param_2'].fillna('missing', inplace=True)
    print_step('Imputation 3/7')
    merge['param_3'].fillna('missing', inplace=True)
    print_step('Imputation 4/7')
    merge['price_missing'] = merge['price'].isna().astype(int)
    merge['price'].fillna(0, inplace=True)
    print_step('Imputation 5/7')
    merge['image_missing'] = merge['image'].isna().astype(int)
    merge['image_top_1'] = merge['image_top_1'].astype('str').fillna('missing')
    print_step('Imputation 6/7')
    merge['description'].fillna('', inplace=True)
    print_step('Imputation 7/7')
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    merge['city'] = merge['city'] + ', ' + merge['region']

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Activation Date')
    merge['activation_date'] = pd.to_datetime(merge['activation_date'])
    merge['day_of_week'] = merge['activation_date'].dt.weekday

    print('~~~~~~~~~~~~~~~~~~~')
    print_step('Basic NLP 1/44')
    merge['num_words_description'] = merge['description'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 2/44')
    merge['num_words_title'] = merge['title'].apply(lambda x: len(str(x).split()))
    print_step('Basic NLP 3/44')
    merge['num_chars_description'] = merge['description'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 4/44')
    merge['num_chars_title'] = merge['title'].apply(lambda x: len(str(x)))
    print_step('Basic NLP 5/44')
    merge['num_capital_description'] = merge['description'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 6/44')
    merge['num_capital_title'] = merge['title'].apply(lambda x: len([c for c in x if c.isupper()]))
    print_step('Basic NLP 7/44')
    merge['num_lowercase_description'] = merge['description'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 8/44')
    merge['num_lowercase_title'] = merge['title'].apply(lambda x: len([c for c in x if c.islower()]))
    print_step('Basic NLP 9/44')
    merge['capital_per_char_description'] = merge['num_capital_description'] / merge['num_chars_description']
    merge['capital_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 10/44')
    merge['capital_per_char_title'] = merge['num_capital_title'] / merge['num_chars_title']
    merge['capital_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 11/44')
    russian_punct = string.punctuation + '—»«„'
    merge['num_punctuations_description'] = merge['description'].apply(lambda x: len([c for c in str(x) if c in russian_punct])) # russian_punct has +0.00001 univariate lift over string.punctuation
    print_step('Basic NLP 12/44')
    merge['punctuation_per_char_description'] = merge['num_punctuations_description'] / merge['num_chars_description']
    merge['punctuation_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 13/44')
    merge['num_punctuations_title'] = merge['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation])) # string.punctuation has +0.0003 univariate lift over russian_punct
    print_step('Basic NLP 14/44')
    merge['punctuation_per_char_title'] = merge['num_punctuations_title'] / merge['num_chars_title']
    merge['punctuation_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 15/44')
    merge['num_words_upper_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print_step('Basic NLP 16/44')
    merge['num_words_lower_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
    print_step('Basic NLP 17/44')
    merge['num_words_entitled_description'] = merge['description'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print_step('Basic NLP 18/44')
    merge['chars_per_word_description'] = merge['num_chars_description'] / merge['num_words_description']
    merge['chars_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 19/44')
    merge['chars_per_word_title'] = merge['num_chars_title'] / merge['num_words_title']
    merge['chars_per_word_title'].fillna(0, inplace=True)
    print_step('Basic NLP 20/44')
    merge['description_words_per_title_words'] = merge['num_words_description'] / merge['num_words_title']
    print_step('Basic NLP 21/44')
    merge['description_words_per_title_words'].fillna(0, inplace=True)
    print_step('Basic NLP 22/44')
    merge['description_chars_per_title_chars'] = merge['num_chars_description'] / merge['num_chars_title']
    print_step('Basic NLP 23/44')
    merge['description_chars_per_title_chars'].fillna(0, inplace=True)
    print_step('Basic NLP 24/44')
    merge['num_english_chars_description'] = merge['description'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 25/44')
    merge['num_english_chars_title'] = merge['title'].apply(lambda ss: len([s for s in ss.lower() if s in string.ascii_lowercase]))
    print_step('Basic NLP 26/44')
    merge['english_chars_per_char_description'] = merge['num_english_chars_description'] / merge['num_chars_description']
    merge['english_chars_per_char_description'].fillna(0, inplace=True)
    print_step('Basic NLP 27/44')
    merge['english_chars_per_char_title'] = merge['num_english_chars_title'] / merge['num_chars_title']
    merge['english_chars_per_char_title'].fillna(0, inplace=True)
    print_step('Basic NLP 28/44')
    merge['num_english_words_description'] = merge['description'].apply(lambda ss: len([w for w in ss.lower().translate(ss.maketrans('', '', russian_punct)).replace('\n', ' ').split(' ') if all([s in string.ascii_lowercase for s in w]) and len(w) > 0]))
    print_step('Basic NLP 29/44')
    merge['english_words_per_word_description'] = merge['num_english_words_description'] / merge['num_words_description']
    merge['english_words_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 30/44')
    merge['max_word_length_description'] = merge['description'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 31/44')
    merge['max_word_length_title'] = merge['title'].apply(lambda ss: np.max([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 32/44')
    merge['mean_word_length_description'] = merge['description'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 33/44')
    merge['mean_word_length_title'] = merge['title'].apply(lambda ss: np.mean([len(w) for w in ss.split(' ')]))
    print_step('Basic NLP 34/44')
    stop_words = {x: 1 for x in stopwords.words('russian')}
    merge['num_stopwords_description'] = merge['description'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    print_step('Basic NLP 35/44')
    merge['number_count_description'] = merge['description'].str.count('[0-9]')
    print_step('Basic NLP 36/44')
    merge['number_count_title'] = merge['title'].str.count('[0-9]')
    print_step('Basic NLP 37/44')
    merge['num_unique_words_description'] = merge['description'].apply(lambda x: len(set(str(x).lower().split())))
    print_step('Basic NLP 38/44')
    merge['unique_words_per_word_description'] = merge['num_unique_words_description'] / merge['num_words_description']
    merge['unique_words_per_word_description'].fillna(0, inplace=True)
    print_step('Basic NLP 39/44')
    merge['sentence'] = merge['description'].apply(lambda x: [s for s in re.split(r'[.!?\n]+', str(x))])
    print_step('Basic NLP 40/44')
    merge['sentence_mean'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.mean(x))
    print_step('Basic NLP 41/44')
    merge['sentence_std'] = merge.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.std(x))
    print_step('Basic NLP 42/44')
    merge['num_sentence'] = merge['sentence'].apply(lambda x: len(x))
    print_step('Basic NLP 43/44')
    merge['words_per_sentence'] = merge['num_words_description'] / merge['num_sentence']
    print_step('Basic NLP 44/44')
    merge.drop(['sentence', 'num_sentence'], axis=1, inplace=True)

    print('~~~~~~~~~~~~')
    print_step('Unmerge')
    dim = train.shape[0]
    train_fe = pd.DataFrame(merge.values[:dim, :], columns = merge.columns)
    test_fe = pd.DataFrame(merge.values[dim:, :], columns = merge.columns)
    print(train_fe.shape)
    print(test_fe.shape)

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('User stats 1/3 1/2')
    train_fe['adjusted_seq_num'] = train_fe['item_seq_number'] - train_fe.groupby('user_id')['item_seq_number'].transform('min')
    print_step('User stats 1/3 2/2')
    test_fe['adjusted_seq_num'] = test_fe['item_seq_number'] - test_fe.groupby('user_id')['item_seq_number'].transform('min')
    print_step('User stats 2/3 1/2')
    train_fe['user_num_days'] = train_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    print_step('User stats 2/3 2/2')
    test_fe['user_num_days'] = test_fe.groupby('user_id')['activation_date'].transform('nunique').astype(int)
    print_step('User stats 3/3 1/5')
    train_fe['date_int'] = pd.to_datetime(train_fe['activation_date']).astype(int)
    test_fe['date_int'] = pd.to_datetime(test_fe['activation_date']).astype(int)
    print_step('User stats 3/3 2/5')
    train_fe['user_days_range'] = train_fe.groupby('user_id')['date_int'].transform('max').astype(int) - train_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    print_step('User stats 3/3 3/5')
    test_fe['user_days_range'] = test_fe.groupby('user_id')['date_int'].transform('max').astype(int) - test_fe.groupby('user_id')['date_int'].transform('min').astype(int)
    print_step('User stats 3/3 4/5')
    train_fe['user_days_range'] = train_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))
    print_step('User stats 3/3 5/5')
    test_fe['user_days_range'] = test_fe['user_days_range'].fillna(0).apply(lambda x: round(x / 10**11))

    print('~~~~~~~~~~~~~~~~~~~~~')
    print_step('Grouping 1/4 1/5')
    train_fe['price'] = train_fe['price'].astype(np.float64)
    test_fe['price'] = test_fe['price'].astype(np.float64)
    print_step('Grouping 1/4 2/5')
    train_enc = train_fe.groupby('category_name')['price'].agg(['mean']).reset_index()
    train_enc.columns = ['category_name', 'cat_price_mean']
    print_step('Grouping 1/4 3/5')
    train_fe = pd.merge(train_fe, train_enc, how='left', on='category_name')
    print_step('Grouping 1/4 4/5')
    test_fe = pd.merge(test_fe, train_enc, how='left', on='category_name')
    print_step('Grouping 1/4 5/5')
    train_fe['cat_price_diff'] = train_fe['price'] - train_fe['cat_price_mean']
    test_fe['cat_price_diff'] = test_fe['price'] - test_fe['cat_price_mean']

    print_step('Grouping 2/4 1/3')
    train_enc = train_fe.groupby('parent_category_name')['parent_category_name'].agg(['count']).reset_index()
    train_enc.columns = ['parent_category_name', 'parent_cat_count']
    print_step('Grouping 2/4 2/3')
    train_fe = pd.merge(train_fe, train_enc, how='left', on='parent_category_name')
    print_step('Grouping 2/4 3/3')
    test_fe = pd.merge(test_fe, train_enc, how='left', on='parent_category_name')

    print_step('Grouping 3/4 1/5')
    train_fe['region_X_cat'] = train_fe['region'] + ':' + train_fe['parent_category_name']
    test_fe['region_X_cat'] = test_fe['region'] + ':' + test_fe['parent_category_name']
    print_step('Grouping 3/4 2/5')
    train_enc = train_fe.groupby('region_X_cat')['region_X_cat'].agg(['count']).reset_index()
    train_enc.columns = ['region_X_cat', 'region_X_cat_count']
    print_step('Grouping 3/4 3/5')
    train_fe = pd.merge(train_fe, train_enc, how='left', on='region_X_cat')
    print_step('Grouping 3/4 4/5')
    test_fe = pd.merge(test_fe, train_enc, how='left', on='region_X_cat')
    print_step('Grouping 3/4 5/5')
    train_fe.drop('region_X_cat', axis=1, inplace=True)
    test_fe.drop('region_X_cat', axis=1, inplace=True)

    print_step('Grouping 4/4 1/4')
    train_enc = train_fe.groupby('city')['city'].agg(['count']).reset_index()
    train_enc.columns = ['city', 'city_count']
    print_step('Grouping 4/4 2/4')
    train_fe = pd.merge(train_fe, train_enc, how='left', on='city')
    print_step('Grouping 4/4 3/4')
    test_fe = pd.merge(test_fe, train_enc, how='left', on='city')
    print_step('Grouping 4/4 4/4')
    test_fe['city_count'].fillna(0, inplace=True)

    print('~~~~~~~~~~~~~')
    print_step('Dropping')
    drops = ['activation_date', 'description', 'title', 'image', 'user_id', 'date_int']
    train_fe.drop(drops, axis=1, inplace=True)
    test_fe.drop(drops, axis=1, inplace=True)

    print('~~~~~~~~~~~')
    print_step('Saving')
    save_in_cache('data_with_fe', train_fe, test_fe)


if not is_in_cache('ohe_data'):
    print('~~~~~~~~~~~~~~~~~~')
    print_step('Cache loading')
    train_fe, test_fe = load_cache('data_with_fe')

    print('~~~~~~~~~~~~')
    print_step('Merging')
    merge = pd.concat([train_fe, test_fe])

    print('~~~~~~~~~~~~~~~~')
    print_step('Dummies 1/2')
    dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1',
                  'day_of_week', 'region', 'city', 'param_1', 'param_2', 'param_3']
    numeric_cols = ['price', 'num_words_description', 'num_words_title', 'num_chars_description',
                    'num_chars_title', 'num_capital_description', 'num_capital_title', 'num_lowercase_title',
                    'capital_per_char_description', 'capital_per_char_title', 'num_punctuations_description',
                    'punctuation_per_char_description', 'punctuation_per_char_title', 'num_words_upper_description',
                    'num_words_lower_description', 'num_words_entitled_description', 'chars_per_word_description',
                    'chars_per_word_title', 'description_words_per_title_words', 'description_chars_per_title_chars',
                    'num_english_chars_description', 'num_english_chars_title', 'english_chars_per_char_description',
                    'english_chars_per_char_title', 'num_english_words_description', 'english_words_per_word_description',
                    'max_word_length_description', 'max_word_length_title', 'mean_word_length_description', 'mean_word_length_title',
                    'num_stopwords_description', 'number_count_description', 'number_count_title', 'num_unique_words_description',
                    'unique_words_per_word_description', 'item_seq_number', 'adjusted_seq_num', 'user_num_days', 'user_days_range',
                    'cat_price_mean', 'cat_price_diff', 'parent_cat_count', 'region_X_cat_count', 'city_count']
    for col in dummy_cols:
        le = LabelEncoder()
        merge[col] = le.fit_transform(merge[col])
    print_step('Scaler')
    scaler = Scaler(columns=numeric_cols)
    train_fe = scaler.fit_transform(train_fe)
    print_step('Dummies 2/2')
    ohe = OneHotEncoder(categorical_features=[merge.columns.get_loc(c) for c in dummy_cols])
    merge = ohe.fit_transform(merge)
    print(merge.shape)

    print('~~~~~~~~~~~')
    print_step('Unmerge')
    merge = merge.tocsr()
    dim = train.shape[0]
    train_ohe = merge[:dim]
    test_ohe = merge[dim:]
    print(train_ohe.shape)
    print(test_ohe.shape)

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('ohe_data', train_ohe, test_ohe)
