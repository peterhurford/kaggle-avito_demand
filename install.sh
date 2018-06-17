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

mkdir ~/.R
export R_LIBS_USER=~/.R
sudo apt install -y r-base-core

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
