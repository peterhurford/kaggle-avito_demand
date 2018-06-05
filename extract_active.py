# HT: https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm

import pandas as pd
import numpy as np

from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


if not is_in_cache('active_feats'):
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print_step('Importing Data 1/5')
    train, test = get_data()
    print_step('Importing Data 2/5')
    train_active = pd.read_csv('train_active.csv')
    print_step('Importing Data 3/5')
    test_active = pd.read_csv('test_active.csv')
    print_step('Importing Data 4/5')
    periods_train = pd.read_csv('periods_train.csv')
    print_step('Importing Data 5/5')
    periods_test = pd.read_csv('periods_test.csv')

    print_step('Merging 1/2 1/3')
    all_samples = pd.concat([
        train,
        train_active,
        test,
        test_active
    ]).reset_index(drop=True)
    print_step('Merging 1/2 2/3')
    all_samples.drop_duplicates(['item_id'], inplace=True)

    print_step('Merging 1/2 3/3')
    all_periods = pd.concat([
        periods_train,
        periods_test
    ])

    print('~~~~~~~~~~~~~~~~~~~~~')
    print_step('Grouping 1/4 1/3')
    all_periods['date_to'] = pd.to_datetime(all_periods['date_to'])
    print_step('Grouping 1/4 2/3')
    all_periods['date_from'] = pd.to_datetime(all_periods['date_from'])
    print_step('Grouping 1/4 3/3')
    all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear


    print_step('Grouping 2/4 1/5')
    gp = all_periods.groupby(['item_id'])[['days_up']]
    gp_df = pd.DataFrame()
    print_step('Grouping 2/4 2/5')
    gp_df['days_up_sum'] = gp.sum()['days_up']
    print_step('Grouping 2/4 3/5')
    gp_df['times_put_up'] = gp.count()['days_up']
    print_step('Grouping 2/4 4/5')
    gp_df.reset_index(inplace=True)
    print_step('Grouping 2/4 5/5')
    gp_df.rename(index=str, columns={'index': 'item_id'})

    print_step('Grouping 3/4 1/4')
    all_periods.drop_duplicates(['item_id'], inplace=True)
    print_step('Grouping 3/4 2/4')
    all_periods = all_periods.merge(gp_df, on='item_id', how='left')
    print_step('Grouping 3/4 3/4')
    all_periods = all_periods.merge(all_samples, on='item_id', how='left')
    print_step('Grouping 3/4 4/4')
    gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
        .rename(index=str, columns={
            'days_up_sum': 'avg_days_up_user',
            'times_put_up': 'avg_times_up_user'
        })
    print_step('Grouping 4/4 1/3')
    n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
        .rename(index=str, columns={
            'item_id': 'n_user_items'
        })
    print_step('Grouping 4/4 2/3')
    gp = gp.merge(n_user_items, on='user_id', how='outer')
    print_step('Grouping 4/4 3/3')
    gp.fillna(0, inplace=True)

    print('~~~~~~~~~~~~~~~~~~~~')
    print_step('Merging 2/2 1/4')
    train = train.merge(gp, on='user_id', how='left')
    print_step('Merging 2/2 2/4')
    test = test.merge(gp, on='user_id', how='left')
    print_step('Merging 2/2 3/4')
    train = train[gp.columns.values]
    print_step('Merging 2/2 4/4')
    test = test[gp.columns.values]

    print('~~~~~~~~~~~~')
    print_step('Caching')
    save_in_cache('active_feats', train, test)
