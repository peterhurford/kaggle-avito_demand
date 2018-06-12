## Solution to the Avito Demand Competition by Peter Hurford, Learnmower, RDizzl3, Sijun He, and Matt Motoki

Additions to the model are done in commits and tracked in CHANGELOG with their impact on the model.

Most things were run on a r4.4xlarge (16 core, 120 GB RAM), some things were run on GPU hardware.


## Installation Instructions

1.) Install kaggle.json

2.) `scp` this repo

3.) Install:

```
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python-pip
sudo apt install -y unzip
sudo apt install -y htop
sudo apt install -y libsm6 libxext6

pip3 install Cython
pip3 install numpy
pip3 install git+https://github.com/anttttti/Wordbatch
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"

wget https://s3.amazonaws.com/avito-demand-kaggle/city_latlons.csv
wget https://s3.amazonaws.com/avito-demand-kaggle/region_macro.csv
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

4.) Optionally set up AWS credentials via `aws configure` and download existing cache files via `python3 sync_cache.py --down`.

5.) Build the features and models. Run in this order:

```
python3 extract_features.py
python3 extract_images.py
python3 extract_active.py
python3 extract_NIMA.py
python3 model_ridge.py
python3 model_fm.py
python3 model_tffm.py
python3 model_cat_bin_ridge.py
python3 model_cat_region_ridge.py
python3 model_parent_cat_ridge.py
python3 model_deep_lgb.py
python3 model_stack_lgb.py
python3 model_lgb_blender.py
```

6.) Optionally upload cache files via `python3 sync_cache.py --up`.
