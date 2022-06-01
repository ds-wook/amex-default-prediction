# amex-default-prediction
### Introduction
This repository is the code that placed ?th [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/overview)

### Benchmark
|Model|CV|Public LB|
|-----|--|------|
|LightGBM(10-stratified kfold - gbdt)|0.789|0.791|
|LightGBM(5-stratified kfold - dart)|-|-|
|CatBoost(10-stratified categorical kfold)|0.789|0.793|
|LightGBM(10-stratified categorical kfold - gbdt)|0.790|0.792|
|**LightGBM(5-stratified categorical kfold - dart)**|**0.792**|**0.794**|

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
--------
Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) & [microsoft recommenders](https://github.com/microsoft/recommenders/tree/main/recommenders).
