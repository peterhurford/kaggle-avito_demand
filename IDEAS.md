- Move embedding into mainline LGB stack, add to stacker models using SVD(20)
- Make a second base LGB with Bayesian encoding and NB features
- Add SVD embedding to blender, also try removing all other SVD
- Switch stacker LGBs to Bayesian encoding if better
- Rotate seeds on models, tune leaves for base model separately
- Try more and different embeddings, SVD on embeddings

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

- price mean and price diff on inception_resnet_v2_top_1

- Add more macroeconomic data (vote Putin)

- Docfreq stats (see https://github.com/Wrosinski/Kaggle-Quora/blob/master/features/Extraction%20-%20Textacy%20Features%2017.05.ipynb)

- POS tagging
- Sentiment analysis

- use mean of log1p for price / diff
- use harmonic mean for price / diff
- use geometric mean for price / diff

- Figure out what to do with Matt's entity embedding on price
- Add Matt's entity embedding on deal probability
- Add Matt's entity embedding on deal threshold
- Add Matt's Price ECDF encoding
- Add Matt's interactions???

- Mean, skew, kurtosis of embeddings

- Similarity encoding (https://dirty-cat.github.io/stable/auto_examples/02_predict_employee_salaries.html#sphx-glr-auto-examples-02-predict-employee-salaries-py)
- Forest Kernels (https://github.com/joshloyal/ForestKernels, https://arxiv.org/abs/1402.4293)

- LGB with CountVectorizer title (see https://www.kaggle.com/him4318/lightgbm-with-aggregated-features-v-2-0)
- Refine embedding LGB, try SVD of multiple embeddings
- Models with poisson loss

- Lower learning rate, more rounds, vary seed
- Tune LGB encoding -- defaults are min_data_per_group=100 max_cat_threshold=32 cat_l2=10.0 cat_smooth=10.0 max_cat_to_onehot=4

- Add RDizzl3's LGB
- Add RDizzle3's Ridge
- Add Sijun's NNs
- Add Matt's NNs
- Add Matt's NB-SVM
- Add Thomas's NNs

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
