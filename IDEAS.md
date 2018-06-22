- Train Poisson models (Running TE on Avito2, Base on Avito1, Ridge on Avito3)
- Add Liu's price level feature and high cardinality category bin feature to models + Thomas's imputation + Liu's image features (TE running on Avito4 -- Need to do for Base, Ridge)

- Try adding Ridge, Poisson and non-Poisson, averaged to blender
- Add Ryan's train_ryan_lgbm_v33.csv/test_ryan_lgbm_v33.csv to blender
- Add other Poisson models from me to blender?
- Train Poisson blender and average

- Add Count lemmas to models
- Add x['level'] = pd.qcut(x['price'], 100, labels=False, duplicates='drop') but cast to categorical

- Add Thomas's / Matt's new price embedding
- Add Sijun's CNNs
- Add Matt and Thomas's RNNs

- Dominant color
- number of colors
- number of key points
- number bright spots <https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/>
- object size <https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/>
- noise, composition, contrast <https://github.com/jeffh/CV-Image-Quality-Analysis>
- https://bitbucket.org/sakoho81/pyimagequalityranking/
- number circles <http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html#hough-circles>
- number lines <http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html#hough-lines>
- number edges <http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny>
- histogram stuff <http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_backprojection/py_histogram_backprojection.html#histogram-backprojection, http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization>
- HarrPSI <http://www.haarpsi.org/software/haarPsi.py, http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf>

- Vet external data

- Mean, skew, kurtosis of embeddings
- Docfreq stats (see https://github.com/Wrosinski/Kaggle-Quora/blob/master/features/Extraction%20-%20Textacy%20Features%2017.05.ipynb)

- POS tagging
- Sentiment analysis

- Rotate seeds on models, tune leaves for base model separately

- Similarity encoding (https://dirty-cat.github.io/stable/auto_examples/02_predict_employee_salaries.html#sphx-glr-auto-examples-02-predict-employee-salaries-py)
- Forest Kernels (https://github.com/joshloyal/ForestKernels, https://arxiv.org/abs/1402.4293)

- Retrain Ridges, FMs, OHE LGB
- Models with poisson loss
- Lower learning rate, more rounds, vary seed
- Tune LGB encoding -- defaults are min_data_per_group=100 max_cat_threshold=32 cat_l2=10.0 cat_smooth=10.0 max_cat_to_onehot=4
- Dart
- XGB
