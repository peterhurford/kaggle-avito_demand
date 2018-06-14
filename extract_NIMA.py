import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
## MobileNet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
## Inception ResNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess
## NASNet
from nasnet import NASNetMobile
from nasnet import preprocess_input as nasnet_preprocess
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd

import argparse

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

parser = argparse.ArgumentParser(description='Extract NIMA features on images')
parser.add_argument('--train', default=False, help='Running on train', action='store_true',)
parser.add_argument('--test', default=False, help='Running on test', action='store_true')
args = parser.parse_args()
if (not args.train) and (not args.test):
    print('Enter at least one of --train or --test, or both') 
if args.train:
    print('Running Training Images')
    ################################################################################
    print('MobileNet')
    base_model = MobileNet((None, None, 3), 
                            alpha=1, include_top=False, 
                            pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    mobileNet_model = Model(base_model.input, x)
    mobileNet_model.load_weights('weights/mobilenet_weights.h5')
    train_datagen = ImageDataGenerator(preprocessing_function = 
        lambda x: mobilenet_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'train',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    mobileNet_scores = mobileNet_model.predict_generator(train_generator, 
                                    verbose = 1, 
                                    use_multiprocessing = True, 
                                    workers = 2)

    mobile_mean = []
    mobile_std = []
    for i in range(mobileNet_scores.shape[0]):
        mobile_mean.append(mean_score(mobileNet_scores[i, :]))
        mobile_std.append(std_score(mobileNet_scores[i, :]))
    mobile_df = pd.DataFrame({'image':train_generator.filenames})
    mobile_df['image'] = mobile_df['image'].apply(lambda x: x[:-4].split('/')[1])
    mobile_df['mobile_mean'] = mobile_mean
    mobile_df['mobile_std'] = mobile_std
    ################################################################################
    print('NasNet')
    base_model = NASNetMobile((224, 224, 3), include_top=False, 
                                pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    nasnet_model = Model(base_model.input, x)
    nasnet_model.load_weights('weights/nasnet_weights.h5')
    train_datagen = ImageDataGenerator(preprocessing_function = 
        lambda x: nasnet_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'train',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    nasnet_scores = nasnet_model.predict_generator(train_generator, 
                                            verbose = 1, 
                                            use_multiprocessing = True, 
                                            workers = 2)
    nasnet_mean = []
    nasnet_std = []
    for i in range(nasnet_scores.shape[0]):
        nasnet_mean.append(mean_score(nasnet_scores[i, :]))
        nasnet_std.append(std_score(nasnet_scores[i, :]))
    nasnet_df = pd.DataFrame({'image':train_generator.filenames})
    nasnet_df['image'] = nasnet_df['image'].apply(lambda x: x[:-4].split('/')[1])
    nasnet_df['nasnet_mean'] = nasnet_mean
    nasnet_df['nasnet_std'] = nasnet_std
    ################################################################################
    base_model = InceptionResNetV2(input_shape=(None, None, 3), 
                                    include_top=False, 
                                    pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    inception_model = Model(base_model.input, x)
    inception_model.load_weights('weights/inception_resnet_weights.h5')
    train_datagen = ImageDataGenerator(
        preprocessing_function = lambda x: inception_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'train',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    inception_scores = inception_model.predict_generator(train_generator, 
                                            verbose = 1, 
                                            use_multiprocessing = True, workers = 2)
    inception_mean = []
    inception_std = []
    for i in range(inception_scores.shape[0]):
        inception_mean.append(mean_score(inception_scores[i, :]))
        inception_std.append(std_score(inception_scores[i, :]))
    inception_df = pd.DataFrame({'image':train_generator.filenames})
    inception_df['image'] = inception_df['image'].apply(lambda x: x[:-4].split('/')[1])
    inception_df['inception_mean'] = inception_mean
    inception_df['inception_std'] = inception_std
    ################################################################################
    print('Joining Results')
    total_df = (mobile_df
            .merge(inception_df, on = 'image')
            .merge(nasnet_df, on = 'image'))
    print('Saving to results to cache/train_img_nima.csv')
    total_df.to_csv('cache/train_img_nima.csv', index = False)
if args.test:
    print('Running Test Images')
    ################################################################################
    print('MobileNet')
    base_model = MobileNet((None, None, 3), 
                            alpha=1, include_top=False, 
                            pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    mobileNet_model = Model(base_model.input, x)
    mobileNet_model.load_weights('weights/mobilenet_weights.h5')
    train_datagen = ImageDataGenerator(preprocessing_function = 
        lambda x: mobilenet_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'test',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    mobileNet_scores = mobileNet_model.predict_generator(train_generator, 
                                    verbose = 1, 
                                    use_multiprocessing = True, 
                                    workers = 2)

    mobile_mean = []
    mobile_std = []
    for i in range(mobileNet_scores.shape[0]):
        mobile_mean.append(mean_score(mobileNet_scores[i, :]))
        mobile_std.append(std_score(mobileNet_scores[i, :]))
    mobile_df = pd.DataFrame({'image':train_generator.filenames})
    mobile_df['image'] = mobile_df['image'].apply(lambda x: x[:-4].split('/')[1])
    mobile_df['mobile_mean'] = mobile_mean
    mobile_df['mobile_std'] = mobile_std
    ################################################################################
    print("NasNet")
    base_model = NASNetMobile((224, 224, 3), include_top=False, 
                                pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    nasnet_model = Model(base_model.input, x)
    nasnet_model.load_weights('weights/nasnet_weights.h5')
    train_datagen = ImageDataGenerator(preprocessing_function = 
        lambda x: nasnet_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'test',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    nasnet_scores = nasnet_model.predict_generator(train_generator, 
                                            verbose = 1, 
                                            use_multiprocessing = True, 
                                            workers = 2)
    nasnet_mean = []
    nasnet_std = []
    for i in range(nasnet_scores.shape[0]):
        nasnet_mean.append(mean_score(nasnet_scores[i, :]))
        nasnet_std.append(std_score(nasnet_scores[i, :]))
    nasnet_df = pd.DataFrame({'image':train_generator.filenames})
    nasnet_df['image'] = nasnet_df['image'].apply(lambda x: x[:-4].split('/')[1])
    nasnet_df['nasnet_mean'] = nasnet_mean
    nasnet_df['nasnet_std'] = nasnet_std
    ################################################################################
    base_model = InceptionResNetV2(input_shape=(None, None, 3), 
                                    include_top=False, 
                                    pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    inception_model = Model(base_model.input, x)
    inception_model.load_weights('weights/inception_resnet_weights.h5')
    train_datagen = ImageDataGenerator(
        preprocessing_function = lambda x: inception_preprocess(np.expand_dims(img_to_array(x), axis=0)))
    train_generator = train_datagen.flow_from_directory(
            'test',
            target_size = (224, 224),
            batch_size=16,
            shuffle=False,
            class_mode=None)
    inception_scores = inception_model.predict_generator(train_generator, 
                                            verbose = 1, 
                                            use_multiprocessing = True, workers = 2)
    inception_mean = []
    inception_std = []
    for i in range(inception_scores.shape[0]):
        inception_mean.append(mean_score(inception_scores[i, :]))
        inception_std.append(std_score(inception_scores[i, :]))
    inception_df = pd.DataFrame({'image':train_generator.filenames})
    inception_df['image'] = inception_df['image'].apply(lambda x: x[:-4].split('/')[1])
    inception_df['inception_mean'] = inception_mean
    inception_df['inception_std'] = inception_std
    ################################################################################
    print('Joining Results')
    total_df = (mobile_df
            .merge(inception_df, on = 'image')
            .merge(nasnet_df, on = 'image'))
    print('Saving to results to cache/test_img_nima.csv')
    total_df.to_csv('cache/test_img_nima.csv', index = False)
