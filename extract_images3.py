# HT: https://www.kaggle.com/bguberfain/vgg16-train-features/code
import os
import gc

import cv2

from math import ceil
from pathlib import PurePath
from scipy import sparse
from threading import Thread
from queue import Queue
from tqdm import tqdm

import numpy as np
import pandas as pd

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from multiprocessing import Value, Pool

from utils import print_step

pool = 'avg' # one of max of avg
batch_size = 64
im_dim = 96
n_channels = 3
limit = None # Limit number of images processed (useful for debug)
resize_mode = 'fit' # One of fit or crop
bar_iterval = 10 # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8) # Used when no image is present

print_step('Loading images')
images_dir = os.path.expanduser('train_jpg')
train_images_dir = os.path.expanduser('train_jpg')
train_image_files = [x.path for x in os.scandir(train_images_dir)]
test_images_dir = os.path.expanduser('test_jpg')
test_image_files = [x.path for x in os.scandir(test_images_dir)]
image_files = train_image_files + test_image_files
image_files = image_files[:1000]
total = len(image_files)
print_step('Loaded %d images' % total)

def generate_files(n_items):
    for item in enumerate(image_files):
        yield item
    
# Based on https://gist.github.com/everilae/9697228
class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator, queue_maxsize):
        self._iterator = iterator
        self._sentinel = object()
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                self._queue.put(value)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()
        

def im_decode(image_path):
    img = cv2.imread(image_path[1])
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    return image_path, img

def predict_batch(model, X_batch):
    X_batch = preprocess_input(np.array(X_batch, dtype=np.float32))
    features_batch = model.predict_on_batch(X_batch)
    features_single = full_model.predict_on_batch(X_batch)
    return [features_batch.astype(np.float16), features_single.astype(np.float16)]
    
    
print_step('Loading models...')
model = InceptionResNetV2(weights='imagenet', pooling=pool, include_top=False)
full_model = InceptionResNetV2(weights='imagenet')

n_items = Value('i', -1)  # Async number of items
features = []
# items_ids = []
pool = Pool(16)
bar = None
X_batch = []
# Threaded generator is usful for both parallel blocking read and to limit
# items buffered by pool.imap (may cause OOM)
generator = ThreadedGenerator( generate_files(n_items), 50 )
for image_path, im in pool.imap(im_decode, generator):
	if bar is None:
		bar = tqdm(total=n_items.value, mininterval=bar_iterval, unit_scale=True)
		
	# Replace None with empty image
	if im is None:
		im = empty_im
		
	X_batch.append(im)
	# items_ids.append(item_id)
	del im

	if len(X_batch) == batch_size:
		preds = predict_batch(model, X_batch)
		feats = [image_path, preds[0], preds[1]]
		features.append(feats)
		del X_batch
		X_batch = []
		bar.update(batch_size)

# Predict last batch
if len(X_batch) > 0:
	preds = predict_batch(model, X_batch)
	feats = [image_path, preds[0], preds[1]]
	features.append(feats)
	bar.update(len(X_batch))

pool.close()
del pool, model, X_batch
bar.close()
gc.collect()

print_step('Concating sparse matrix...')
import pdb
pdb.set_trace()
features = sparse.vstack(features)

print_step('Saving sparse matrix...')
sparse.save_npz('features.npz', features, compressed=True)
