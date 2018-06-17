import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from utils import print_step


def run_cv_model(train, test, target, model_fn, params, eval_fn, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=2017)
    fold_splits = kf.split(train)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros(train.shape[0])
    i = 1
    for dev_index, val_index in fold_splits:
        print_step('Started ' + label + ' fold ' + str(i) + '/5')
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.values[dev_index], train.values[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
            dev_X = pd.DataFrame(dev_X, columns=train.columns)
            val_X = pd.DataFrame(val_X, columns=train.columns)
            for (column, dtype) in list(zip(train.columns, list(train.dtypes))):
                dev_X[column] = dev_X[column].astype(dtype)
                val_X[column] = val_X[column].astype(dtype)
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        cv_score = eval_fn(val_y, pred_val_y)
        cv_scores.append(eval_fn(val_y, pred_val_y))
        print_step(label + ' cv score ' + str(i) + ' : ' + str(cv_score))
        i += 1
    print_step(label + ' cv scores : ' + str(cv_scores))
    print_step(label + ' mean cv score : ' + str(np.mean(cv_scores)))
    print_step(label + ' std cv score : ' + str(np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores}
    return results
