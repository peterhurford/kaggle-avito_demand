import os
import argparse

parser = argparse.ArgumentParser(description='Sync the local cache with S3.')
parser.add_argument('--up', default=False, action='store_true',
                    help='Upload data in the cache to S3.')
parser.add_argument('--down', default=False, action='store_true',
                    help='Download data in the cache from S3.')
args = parser.parse_args()
is_up_action = (args.up == True)
is_down_action = (args.down == True)

def up():
    if os.path.isdir('cache'):
        os.system('aws s3 cp --recursive cache s3://avito-demand-kaggle-private')
    else:
        raise ValueError('Could not find cache directory.')

def down():
    os.system('aws s3 cp --recursive s3://avito-demand-kaggle-private cache')

if is_up_action and is_down_action:
    up()
    down()
elif is_up_action:
    up()
elif is_down_action:
    down()
