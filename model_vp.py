import random
import string
import os.path

from nltk.corpus import stopwords

from datetime import datetime

import pandas as pd
import numpy as np

import pathos.multiprocessing as mp

from sklearn.model_selection import train_test_split

from vowpal_platypus import run
from vowpal_platypus.models import ftrl, linear_regression
from vowpal_platypus.utils import clean

from cv import run_cv_model
from utils import rmse, print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


def process_line(item, header, predict=False):
    headers = map(clean, header.split(','))
    items = dict(zip(headers, item.replace('\n', '').split(',')))
    if items.get('item_id'):
        items.pop('item_id')
    if items.get('user_id'):
        items.pop('user_id')
    if items.get('image'):
        items.pop('image')
    if not predict:
        target = float(items.pop('deal_probability'))
    items['activation_date'] = pd.to_datetime(items['activation_date']).dayofweek
    if items['price'] != '':
        items['price'] = np.log1p(float(items['price']))
    else:
        items['price'] = 0
    if items['price'] > 9:
        items['price'] = 9
    items['image_top_1'] = 'img' + str(items['image_top_1'])
    items['item_seq_number'] = 'seq' + str(items['item_seq_number'])
    items['activation_date'] = 'day' + str(items['activation_date'])

    parent_cat = 'parent_' + items['parent_category_name'].replace(' ', '_')
    cat = 'cat_' + items['parent_category_name'].replace(' ', '_') + '_' + items['category_name'].replace(' ', '_')
    p1 = 'p1_' + items['parent_category_name'].replace(' ', '_') + '_' + items['category_name'].replace(' ', '_') + '_' + items['param_1'].replace(' ', '_')
    p2 = 'p2_' + items['parent_category_name'].replace(' ', '_') + '_' + items['category_name'].replace(' ', '_') + '_' + items['param_1'].replace(' ', '_') + '_' + items['param_2'].replace(' ', '_')
    p3 = 'p3_' + items['parent_category_name'].replace(' ', '_') + '_' + items['category_name'].replace(' ', '_') + '_' + items['param_1'].replace(' ', '_') + '_' + items['param_2'].replace(' ', '_') + '_' + items['param_3'].replace(' ', '_')
    if items['param_3'] != '':
        cats = [parent_cat, cat, p1, p2, p3]
    else:
        cats = [parent_cat, cat, p1, p2]
    stop_words = stopwords.words('russian')
    for cat in ['title', 'description', 'param_1', 'param_2', 'param_3', 'category_name', 'parent_category_name']:
        for s in string.punctuation:
            items[cat] = str(items[cat]).replace(s, '')
        items[cat] = filter(lambda s: s not in stop_words,
                            unicode(items[cat], 'utf8').lower().split(' '))

        if cat == 'title':
            items[cat] = map(lambda s: u'word_' + s[0:6], items[cat])
        else:
            items[cat] = map(lambda s: u'word_' + s, items[cat])
        items[cat] = filter(lambda s: s != u'word_', items[cat])
        items[cat] = ' '.join(items[cat]).encode('utf-8').strip()

    # names = {'a': items['activation_date'],
    #          'c': ' '.join(cats),
    #          'i': items['image_top_1'],
    #          'p': items['price'],
    #          'u': items['user_type'],
    #          'r': items['region'].replace(' ', '_') + ' ' + items['city'].replace(' ', '_') + '_' + items['region'].replace(' ', '_'),
    #          's': items['item_seq_number'],
    #          't': (' '.join(items['parent_category_name']) + ' ' + ' '.join(items['category_name']) + ' ' + ' '.join(items['param_1']) + ' ' + ' '.join(items['param_2']) + ' ' + ' '.join(items['param_3']) + ' ' +  ' '.join(items['title'])).replace('  ', ' ')}
    names = {'t': (items['parent_category_name'] + ' ' + items['category_name'] + ' ' + items['param_1'] + ' ' + items['param_2'] + ' ' + items['param_3'] + ' ' + items['title']).replace('  ', ' ')}
    if not predict:
        names['label'] = target
    return names


def run_vp(train_X, train_y, test_X, test_y, test_X2):
    print_step('Munging 1/8')
    train_X['description'] = train_X['description'].str.replace('\n', ' ')
    test_X['description'] = test_X['description'].str.replace('\n', ' ')
    test_X2['description'] = test_X2['description'].str.replace('\n', ' ')
    print_step('Munging 2/8')
    train_X['description'] = train_X['description'].str.replace(',', ' ')
    test_X['description'] = test_X['description'].str.replace(',', ' ')
    test_X2['description'] = test_X2['description'].str.replace(',', ' ')
    print_step('Munging 3/8')
    train_X['category_name'] = train_X['category_name'].str.replace(',', ' ')
    test_X['category_name'] = test_X['category_name'].str.replace(',', ' ')
    test_X2['category_name'] = test_X2['category_name'].str.replace(',', ' ')
    print_step('Munging 4/8')
    train_X['param_1'] = train_X['param_1'].str.replace(',', ' ')
    test_X['param_1'] = test_X['param_1'].str.replace(',', ' ')
    test_X2['param_1'] = test_X2['param_1'].str.replace(',', ' ')
    print_step('Munging 5/8')
    train_X['param_2'] = train_X['param_2'].str.replace(',', ' ')
    test_X['param_2'] = test_X['param_2'].str.replace(',', ' ')
    test_X2['param_2'] = test_X2['param_2'].str.replace(',', ' ')
    print_step('Munging 6/8')
    train_X['param_3'] = train_X['param_3'].str.replace(',', ' ')
    test_X['param_3'] = test_X['param_3'].str.replace(',', ' ')
    test_X2['param_3'] = test_X2['param_3'].str.replace(',', ' ')
    print_step('Munging 7/8')
    train_X['title'] = train_X['title'].str.replace(',', ' ')
    test_X['title'] = test_X['title'].str.replace(',', ' ')
    test_X2['title'] = test_X2['title'].str.replace(',', ' ')
    print_step('Munging 8/8')
    train_X['description'].fillna('missing', inplace=True)
    test_X['description'].fillna('missing', inplace=True)
    test_X2['description'].fillna('missing', inplace=True)

    print_step('Writing 1/3')
    train_X['deal_probability'] = train_y.values
    rando = list(string.lowercase)
    random.shuffle(rando)
    rando = ''.join(rando[:10])
    train_filename = 'vp_tmp_train_' + rando + '.csv'
    train_X.to_csv(train_filename, index=False)
    print_step('Writing 2/3')
    test_filename = 'vp_tmp_test_' + rando + '.csv'
    test_X.to_csv(test_filename, index=False)
    print_step('Writing 3/3')
    test_filename2 = 'vp_tmp_test2_' + rando + '.csv'
    test_X2.to_csv(test_filename2, index=False)

    print_step('VP Define Model')
    model = linear_regression(name='AvitoVP',
                              passes=20,
                              bits=24,
                              l1=0.0,
                              l2=0.000226,
                              decay_learning_rate=0.99,
                              learning_rate=0.1,
                              ngram='t2',
                              debug=True,
                              debug_rate=100000,
                              cores=1)

    with open(train_filename) as f:
        header = f.readline()
    print_step('VP Train and Predict')
    pred_test_y = run(model,
                      train_filename=train_filename,
                      predict_filename=test_filename,
                      train_line_function=lambda line: process_line(line, header),
                      predict_line_function=lambda line: process_line(line, header, predict=True),
                      clean=False)
    print_step('VP Predict 2/2')
    pred_test_y2 = run(model,
                       train_filename=None,
                       predict_filename=test_filename,
                       predict_line_function=lambda line: process_line(line, header, predict=True),
                       clean=True)
    return np.array(pred_test_y), np.array(pred_test_y2)

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

print('~~~~~~~~~~~')
print_step('Run VP')
results = run_cv_model(train, test, target, run_vp, rmse, 'vp')
import pdb
pdb.set_trace()

print('~~~~~~~~~~')
print_step('Cache')
save_in_cache('vp_model', pd.DataFrame({'vp': results['train']}),
                          pd.DataFrame({'vp': results['test']}))
# [2018-05-30 20:09:29.468592] vp cv scores : [0.23800563305533942, 0.23743390079098547, 0.2368078848924554, 0.23673838455450671, 0.236930845325303]
# [2018-05-30 20:09:29.468927] vp mean cv score : 0.23718332972371797
# [2018-05-30 20:09:29.470959] vp std cv score : 0.0004778393088301725
