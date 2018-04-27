from datetime import datetime
from math import sqrt

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, roc_auc_score


def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def univariate_analysis(target, feature):
    score = roc_auc_score(target > 0, feature)
    return 1 - score if score < 0.5 else score


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
