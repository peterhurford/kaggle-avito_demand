import argparse
import os

import numpy as np
import pandas as pd

from PIL import Image

from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import keras.applications.vgg16 as vgg16
import keras.applications.inception_resnet_v2 as inception_resnet_v2
import keras.applications.densenet as densenet

from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


parser = argparse.ArgumentParser()
parser.add_argument('--start')
parser.add_argument('--stop')
args = parser.parse_args()
start = int(args.start)
stop = int(args.stop)


# HT: https://www.kaggle.com/wesamelshamy/high-correlation-feature-image-classification-conf
def image_classify(model, pak, img, top_n=3, headless=False):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    if headless:
        return preds
    else:
        return pak.decode_predictions(preds, top=top_n)[0]


def classify(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    try:
        img = Image.open(image_path)
        inception_resnet_preds = image_classify(inception_resnet_model, inception_resnet_v2, img, top_n=1)
        return [image_path.replace('train_jpg/', '').replace('test_jpg/', ''), inception_resnet_preds[0][1], inception_resnet_preds[0][2]]
    except OSError:
        return None


if not is_in_cache('inception_' + str(start) + '_' + str(stop)):
    print_step('Initializing models')
    inception_resnet_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet')

    print_step('Loading images')
    images_dir = os.path.expanduser('train_jpg')
    train_images_dir = os.path.expanduser('train_jpg')
    train_image_files = [x.path for x in os.scandir(train_images_dir)]
    test_images_dir = os.path.expanduser('test_jpg')
    test_image_files = [x.path for x in os.scandir(test_images_dir)]
    image_files = train_image_files + test_image_files
    print_step('Loaded %d images - iterating between %d and %d' % (len(image_files), start, stop))
    image_files = image_files[start:stop]

    data = []
    total = len(image_files)
    i = start
    for image_file in image_files:
        print_step('%d / %d' % (i, stop))
        res = classify(image_file)
        i += 1
        if res:
            data += [res]

    save_in_cache('inception_' + str(start) + '_' + str(stop), pd.DataFrame(data, columns = ['image', 'inception_label', 'inception_pred']), None)
else:
    print('%d - %d Already run!' % (start, stop))
