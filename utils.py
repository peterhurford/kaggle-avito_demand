from datetime import datetime
from math import sqrt

import pandas as pd

from sklearn.metrics import mean_squared_error


def print_step(step):
    print('[{}]'.format(datetime.now()) + ' ' + step)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
