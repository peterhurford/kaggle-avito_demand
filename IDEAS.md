- Add lat / lon to blender
- Add SVD and TFIDF stats to base LGB, and add to blender

- Docfreq stats (see https://github.com/Wrosinski/Kaggle-Quora/blob/master/features/Extraction%20-%20Textacy%20Features%2017.05.ipynb)

- Dominant color
- average, standard deviation and minimum of Brightness in HSV color space
- average and standard deviation of Saturation in HSV color space
- average, standard deviation and minimum of Luminance in YUV color space
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

- POS tagging
- Sentiment analysis

- Add more macroeconomic data

- Similarity encoding (https://dirty-cat.github.io/stable/auto_examples/02_predict_employee_salaries.html#sphx-glr-auto-examples-02-predict-employee-salaries-py)
- Forest Kernels (https://github.com/joshloyal/ForestKernels, https://arxiv.org/abs/1402.4293)

- Handle price outliers / Try to predict price

- skew and kurtosis of sent2vec (see https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py)

- LGB with CountVectorizer title (see https://www.kaggle.com/him4318/lightgbm-with-aggregated-features-v-2-0)
- Make embedding LGBs
- Models with poisson loss

- Lower learning rate, more rounds, vary seed
- Tune LGB encoding -- defaults are min_data_per_group=100 max_cat_threshold=32 cat_l2=10.0 cat_smooth=10.0 max_cat_to_onehot=4

- Add RDizzl3 LGB
- Add RDizzle3 Ridge
- Add NNs
- Image labeling confidence
- Add Entity embedding

- Dart
- XGB
- RF


PATH TO 0.2166
- Improving NN (0.0001)
- Adding new NNs (0.0002)
- RDizzl3 models (0.0002)
- Embedding LGBs (0.0002)
- Model tuning (0.0001)
- Stacking and foresting (0.0001)
- Better features (0.0001)
