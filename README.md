Run on a r4.4xlarge (16 core, 120 GB RAM)

1.) Install kaggle.json

2.) Upload `city_latlons.csv` and `region_macro.csv`

3.) To run:

```
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y unzip
sudo apt install -y htop

pip3 install Cython
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"

kaggle competitions download -c avito-demand-prediction -f train.csv.zip
kaggle competitions download -c avito-demand-prediction -f test.csv.zip
mv ~/.kaggle/competitions/avito-demand-prediction/*.zip .
unzip train.csv
unzip test.csv

mkdir cache

python3 features.py
python3 ridge.py
python3 lgb.py
```
