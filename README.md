## Solution to the Avito Demand Competition by Peter Hurford, Learnmower, and RDizzl3

Additions to the model are done in commits and tracked in CHANGELOG with their impact on the model.

Everything so far is run on a r4.4xlarge (16 core, 120 GB RAM)


## Installation Instructions

1.) Install kaggle.json

2.) Upload [`city_latlons.csv`](https://s3.amazonaws.com/avito-demand-kaggle/city_latlons.csv) and [`region_macro.csv`](https://s3.amazonaws.com/avito-demand-kaggle/region_macro.csv).

3.) Install:

```
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python-pip
sudo apt install -y unzip
sudo apt install -y htop
sudo apt install -y libsm6 libxext6

pip3 install Cython
pip3 install -r requirements.txt
pip install Cython
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"

sudo apt-get -y install build-essential clang-3.5 llvm
sudo ln -s /usr/bin/clang-3.5 /usr/bin/clang; sudo ln -s /usr/bin/clang++-3.5 /usr/bin/clang++
sudo apt-get -y install libffi-dev libssl-dev libxml2-dev libxslt1-dev libjpeg8-dev zlib1g-dev
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install default-jre
git clone https://github.com/JohnLangford/vowpal_wabbit.git && cd vowpal_wabbit
make && sudo make install && cd ..
pip install git+https://github.com/peterhurford/vowpal_platypus.git@v2.2
pip install retrying

kaggle competitions download -c avito-demand-prediction -p .
unzip train.csv
unzip test.csv
unzip train_jpg.zip
mv data/competition_files/train_jpg/ .
unzip test_jpg.zip
mv data/competition_files/test_jpg/ .
unzip train_active.csv
unzip test_active.csv
unzip periods_train.csv
unzip periods_test.csv
rm *.zip
rm -rf data

mkdir cache
mkdir submit
```

4.) Set up AWS credentials via `aws configure`.

5.) Optionally download existing cache files via `python3 sync_cache.py --down`.

6.) Run:

```
python3 extract_features.py
python3 extract_images.py
python3 extract_active.py
python3 model_ridge.py
python3 model_cat_bin_ridge.py
python3 model_cat_region_ridge.py
python3 model_parent_cat_ridge.py
python3 model_ridge_lgb.py
```

7.) Optionally upload cache files via `python3 sync_cache.py --up`.
