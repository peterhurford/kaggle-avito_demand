import re, regex
import string
import pymorphy2

from math import sqrt
from datetime import datetime

from multiprocessing import Pool
import pathos.multiprocessing as mp

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def univariate_analysis(target, feature):
    score = roc_auc_score(target > 0, feature)
    return 1 - score if score < 0.5 else score


stop_words = stopwords.words('russian')

def normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stop_words)


def clean_text(text):
    text = bytes(text, encoding="utf-8")
    text = text.lower()
    text = re.sub(b'(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', b' ', text)
    text = re.sub(b'\s+(?=\d)|(?<=\d)\s+', b' ', text)
    text = text.replace(b"\b", b" ")
    text = text.replace(b"\r", b" ")
    text = regex.sub(b"\s+", b" ", text)
    text = str(text, 'utf-8')
    text = re.sub(r"\W+", " ", text.lower())
    return text


# https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
class Scaler(BaseEstimator, TransformerMixin): 
    def __init__(self, columns, copy=True, feature_range=(0, 1)):
        self.scalers = [MinMaxScaler(feature_range=feature_range, copy=copy)]
        self.columns = columns

    def fit(self, X, y=None):
        for scaler in self.scalers:
            scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = X[self.columns]
        for scaler in self.scalers:
            X_scaled = pd.DataFrame(scaler.transform(X_scaled), columns=self.columns, index=X.index)
        X_not_scaled = X[list(set(init_col_order) - set(self.columns))]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False, calc_std=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original
        self.calc_std = calc_std

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if not self.calc_std:
                if self.keep_original:
                    train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target, 'mean')
                else:
                    train[col], test[col] = self.encode_column(train[col], test[col], target, 'mean')
            else:
                train[col + '_te_mean'], test[col + '_te_mean'] = self.encode_column(train[col], test[col], target, 'mean')
                train[col + '_te_std'], test[col + '_te_std'] = self.encode_column(train[col], test[col], target, 'std')
                if not self.keep_original:
                    train.drop(col, axis=1, inplace=True)
                    test.drop(col, axis=1, inplace=True)
        return train, test

    def encode_column(self, trn_series, tst_series, target, function):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg([function, 'count'])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages['count'] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        if function == 'mean':
            prior = target.mean()
        elif function == 'std':
            prior = target.std()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages[function] * smoothing
        averages.drop([function, 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_' + function).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_' + function).fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)


def bin_and_ohe_data(train, test, numeric_cols=None, dummy_cols=None, nbins=4):
    train_ohe = None
    test_ohe = None
    if numeric_cols:
        print_step('Scaling numerics')
        scaler = Scaler(columns=numeric_cols)
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        print_step('Binning numerics')
        for col in numeric_cols:
            print(col)
            train[col] = pd.qcut(train[col], nbins, labels=False, duplicates='drop')
            test[col] = pd.qcut(test[col], nbins, labels=False, duplicates='drop')

    print_step('Dummies')
    for col in numeric_cols + dummy_cols:
        print(col)
        lb = LabelBinarizer(sparse_output=True)
        if train_ohe is not None:
            train_ohe = hstack((train_ohe, lb.fit_transform(train[col].fillna('').astype('str')))).tocsr()
            print(train_ohe.shape)
            test_ohe = hstack((test_ohe, lb.transform(test[col].fillna('').astype('str')))).tocsr()
            print(test_ohe.shape)
        else:
            train_ohe = lb.fit_transform(train[col].fillna('').astype('str')).tocsr()
            print(train_ohe.shape)
            test_ohe = lb.transform(test[col].fillna('').astype('str')).tocsr()
            print(test_ohe.shape)
    return train_ohe, test_ohe
