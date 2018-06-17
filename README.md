## Solution to the Avito Demand Competition by Peter Hurford, Learnmower, RDizzl3, Sijun He, and Matt Motoki

Additions to the model are done in commits and tracked in CHANGELOG with their impact on the model.

Most things were run on a r4.4xlarge (16 core, 120 GB RAM), some things were run on GPU hardware.


## Installation Instructions

1.) Install kaggle.json

2.) `scp` this repo

3.) Install: Run `install.sh`

4.) Optionally set up AWS credentials via `aws configure` and download existing cache files via `python3 sync_cache.py --down`.

5.) Build the features and models. Run in this order:

```
python3 extract_features.py
python3 extract_images.py
python3 extract_active.py
python3 extract_NIMA.py --train --test
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
