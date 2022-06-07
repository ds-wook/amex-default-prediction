# amex-default-prediction
### Introduction
This repository is the code that placed ?th [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/overview)

### Benchmark
|Model|CV|Public LB|
|-----|--|------|
|XGBoost(10-KFold - gbdt)|0.792|0.793|
|LightGBM(10-stratified kfold - gbdt)|0.789|0.791|
|CatBoost(10-stratified kfold)|0.789|0.793|
|CatBoost(10-stratified kfold) - trick|0.7927|0.793|
|CatBoost(10-stratified categorical kfold) - trick|0.790|0.791|
|LightGBM(10-stratified kfold - gbdt)|0.790|0.792|
|LightGBM(10-stratified kfold - dart)|0.7926|0.795|
|**LightGBM(10-stratified kfold - dart) - trick**|**0.7931**|**0.796**|
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
    |   ├── __init__.py    <- Makes src a Python module
    │   └── dataset.py
    │
    ├── features           <- Scripts of feature engineering
    │   ├── __init__.py
    |   ├── build.py
    |   └── select.py
    |
    ├── models             <- build train models
    │   ├── __init__.py
    |   ├── base.py
    |   ├── boosting.py
    |   └── infer.py
    |
    ├── tuning             <- tuning models by optuna
    │   ├── __init__.py
    |   ├── base.py
    |   └── boosting.py
    │
    └── utils              <- utils files
        ├── __init__.py
        └── utils.py
```

### Reference
+ [DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
+ [rounding-trick](https://www.kaggle.com/code/jiweiliu/amex-catboost-rounding-trick)

--------
Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) & [microsoft recommenders](https://github.com/microsoft/recommenders/tree/main/recommenders).
