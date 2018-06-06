import logging
import sys

import pandas as pd
import numpy as np

from pic2vec import ImageFeaturizer

from cache import get_data
from utils import print_step


# print('~~~~~~~~~~~~~~~~~~~')
# print_step('Importing Data')
# train, test = get_data()

# print('~~~~~~~~~~~~~~~~~~~')
# print_step('Processing 1/3')
# trainp = train[['item_id', 'deal_probability', 'image']]
# testp = test[['item_id', 'image']]
# print_step('Processing 2/3')
# trainp.columns = ['item_id', 'label', 'images']
# testp.columns = ['item_id', 'images']
# print_step('Processing 3/3')
# trainp['images'] = trainp['images'].apply(lambda s: str(s) + '.jpg' if str(s) != 'nan' else '')
# testp['images'] = testp['images'].apply(lambda s: str(s) + '.jpg' if str(s) != 'nan' else '')
# trainp = trainp.head(1000)
# trainp['images'] = trainp['images'].astype(str)
# testp['images'] = testp['images'].astype(str)

# print('~~~~~~~~~~~~~~~~')
# print_step('Writing 1/2')
# trainp.to_csv('train_pic2vec.csv', index=False)
# print_step('Writing 2/2')
# testp.to_csv('test_pic2vec.csv', index=False)


root = logging.getLogger()
root.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


print('~~~~~~~~~~~~~~~~~~~~')
print_step('Featurizing 1/2')
featurizer = ImageFeaturizer(depth=1, auto_sample=False, model='squeezenet')
print_step('Featurizing 2/2')
image_path = 'train_jpg'
csv_path = 'train_pic2vec.csv'
featurizer.featurize(batch_size=10000, save_features=True, image_column_headers=['images'], image_path = image_path)
import pdb
pdb.set_trace()
