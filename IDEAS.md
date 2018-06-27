- Tune LGB encoding -- defaults are min_data_per_group=100 max_cat_threshold=32 cat_l2=10.0 cat_smooth=10.0 max_cat_to_onehot=4
- Rotate seeds on models, tune leaves for base model separately
- Retrain models
- Lower learning rate, more rounds, vary seed

- Dart
- XGB
- Train model with TFIDF and all features
- Train flat model with all features and all single models, use 31 leaves
- Foresting


- Similarity encoding (https://dirty-cat.github.io/stable/auto_examples/02_predict_employee_salaries.html#sphx-glr-auto-examples-02-predict-employee-salaries-py)
- Forest Kernels (https://github.com/joshloyal/ForestKernels, https://arxiv.org/abs/1402.4293)
