# amex-default-prediction
### Introduction
This repository is the code that placed ?th [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/overview)

### Benchmark
|Model|CV|Public LB|
|-----|--|------|
|TabNet(10-stratified kfold)|0.789|0.790|
|XGBoost(10-KFold - gbdt)|0.792|0.793|
|XGBoost(10-KFold - gbdt) - trick|0.793|0.794|
|LightGBM(10-stratified kfold - gbdt)|0.789|0.791|
|CatBoost(10-stratified kfold)|0.789|0.793|
|CatBoost(10-stratified kfold) - trick|0.7927|0.793|
|CatBoost(10-stratified categorical kfold) - trick|0.790|0.791|
|LightGBM(10-stratified kfold - gbdt)|0.790|0.792|
|LightGBM(10-stratified kfold - dart)|0.7926|0.795|
|LightGBM(10-stratified kfold pay-features - all - dart) - trick|0.7961|**0.797**|
|LightGBM(5-stratified kfold - dart) - trick|0.7960|**0.797**|
|LightGBM(5-stratified kfold time-features - shap - dart) - trick|**0.7970**|**0.797**|
### Project Organization
```
├── LICENSE
├── README.md
├── config                 <- config yaml files
│
├── res
|   ├── data               <- encoding pickle file
|   ├── models             <- Trained and serialized models
|
├── notebooks              <- ipykernel
│
└── src                    <- Source code for use in this project
    │
    ├── data               <- Scripts to preprocess data
    │   └── dataset.py
    │
    ├── features           <- Scripts of feature engineering
    |   ├── build.py
    |   └── select.py
    |
    ├── models             <- build train models
    |   ├── base.py
    |   ├── boosting.py
    |   ├── infer.py
    |   ├── network.py
    |   └── stacking.py
    |
    ├── tuning             <- tuning models by optuna
    |   ├── base.py
    |   └── boosting.py
    │
    └── utils              <- utils files
        └── utils.py
```

### Reference
+ [DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
+ [rounding-trick](https://www.kaggle.com/code/jiweiliu/amex-catboost-rounding-trick)
+ [pay-features](https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb)
+ [Ensemble model](https://www.sciencedirect.com/science/article/pii/S0957417421003407)
+ [feature-engineering](https://www.kaggle.com/code/susnato/amex-data-preprocesing-feature-engineering)
--------
Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) & [microsoft recommenders](https://github.com/microsoft/recommenders/tree/main/recommenders).
