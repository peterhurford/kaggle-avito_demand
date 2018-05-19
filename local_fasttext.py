import os
import fasttext

import pandas as pd

from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


print_step('Concatenating train and test 1/5')
train, test = get_data()
print_step('Concatenating train and test 2/5')
train['desc'] = train['title'] + ' ' + train['description'].fillna('')
print_step('Concatenating train and test 3/5')
test['desc'] = test['title'] + ' ' + test['description'].fillna('')
print_step('Concatenating train and test 4/5')
train.drop([c for c in train.columns if c != 'desc'], axis=1, inplace=True)
test.drop([c for c in test.columns if c != 'desc'], axis=1, inplace=True)
print_step('Concatenating train and test 5/5')
merged = pd.concat([train, test])

print_step('Writing to disk')
merged.to_csv('merged.csv', index=False)

print_step('Training local FastText model')
model = fasttext.skipgram('merged.csv', 'cache/local_fasttext_model', min_count=1)

print_step('Cleaning')
os.remove('merged.csv')
