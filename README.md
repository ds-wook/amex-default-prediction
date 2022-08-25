# American Express - Default Prediction

### Model
My model's name is **EAA(Ensemble is Always Answer)**.
|Architecture|
|-------------------|
|![architecture](https://user-images.githubusercontent.com/46340424/186546573-ab0b711c-6d65-4bd3-8c11-39a681f42143.png)|

Among the boosting models, the LightGBM DART model performed the best. Due to the high noise of the metric, it is slow but can guarantee performance
I decided to use a DART model. Meanwhile, in the case of Catboost, categorical features were learned by adding first features.
TabNet is a type of neural network, and in order to secure the diversity of the GBDT model, an ensemble was attempted by giving a small weight value even if the performance was poor.
The LightGBMs learned with various features were all stacked with XGBoost.

### Feature Engineering
1. Create lag features through time features
2. statement feature: Check the customer's statement (SDist)
3. First feature and Last feature is importance features

### Seed
Performance difference was severe depending on the seed value due to noise of the metric.
Therefore, after learning several seeds, we try to find the optimal cv value.
And they all found and ensemble weight in a gradient way.

### Ensemble Method
```python
def get_best_weights(oofs: List[np.ndarray], target: np.ndarray) -> np.ndarray:
    """
    Get best weights
    Args:
        oofs: oofs of models
        target: target of train data
    Returns:
        best weights
    """
    weight_list = []
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])

    logging.info("Blending Start")
    kf = KFold(n_splits=5)
    for fold, (train_idx, _) in enumerate(kf.split(oofs[0]), 1):
        res = minimize(
            get_score,
            weights,
            args=(train_idx, oofs, target),
            method="Nelder-Mead",
            tol=1e-06,
        )
        logging.info(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    mean_weight = np.insert(mean_weight, len(mean_weight), 1 - np.sum(mean_weight))
    logging.info(f"optimized weight: {mean_weight}\n")

    return mean_weight
```
I use weight optimization of ensemble method. Using the KFold method, find the weight as the average value by using the Nelder and Mead method.

### Benchmark
|Model|CV|Public LB|Private LB|
|-----|--|------|---------|
|XGBoost(10-KFold - gbdt)|0.792|0.793|-|
|TabNet(10-StratifiedKFold)|0.789|0.790|-|
|CatBoost(5-StratifiedKFold sdist-lag-features - dart) - seed22|0.7953|0.797|-|
|CatBoost(5-StratifiedKFold sdist-lag-features - dart) - seed42|0.7954|0.797|-|
|CatBoost(5-StratifiedKFold sdist-lag-features - dart) - seed99|0.7958|0.797|-|
|CatBoost(5-StratifiedKFold sdist-lag-features - dart) - seed3407|0.7948|0.797|-|
|LightGBM(5-StratifiedKFold time-features - shap - dart) - trick|0.7970|0.797|-|
|LightGBM(5-StratifiedKFold time-lag-features - dart) - trick|0.7973|0.797|-|
|LightGBM(5-StratifiedKFold diff-features - dart) - trick|0.7973|0.799|-|
|LightGBM(5-StratifiedKFold trick-features - dart) - seed42|0.7977|0.798|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed22|**0.7981**|0.798|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed42|0.7979|0.798|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed88|0.7977|0.799|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed94|0.7972|0.799|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed99|0.7979|0.799|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed2020|0.7978|0.798|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed2222|0.7976|0.799|-|
|LightGBM(5-StratifiedKFold sdist-features - dart) - seed3407|0.7977|0.799|-|
|LightGBM(5-StratifiedKFold sdist-lag-features - dart) - seed3407|0.7977|0.799|-|
|LightGBM(5-StratifiedKFold bruteforce-features - dart) - seed22|0.7978|0.799|-|
|LightGBM(5-StratifiedKFold bruteforce-features - dart) - seed42|**0.7981**|**0.799**|-|
|LightGBM(5-StratifiedKFold bruteforce-features - dart) - seed99|0.7979|0.799|-|
|LightGBM(5-StratifiedKFold bruteforce-features - dart) - seed3407|0.7978|0.799|-|
|LightGBM(5-StratifiedKFold sdist-lag-features - dart) - seed5230|0.7963|0.799|-|
|XGBoost(10-KFold - stacking regression)|0.7985|0.799|-|
|Ensemble is Always Answer|**0.79952**|**0.799**|-|

### Project Organization
```
├── LICENSE
├── README.md
├── config                 <- config yaml files
│
├── res
|   ├── data               <- encoding pickle file
|   └── models             <- Trained and serialized models
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
    |   ├── callbacks.py   
    |   ├── infer.py
    |   └── network.py
    |
    ├── tuning             <- tuning models by optuna
    |   ├── base.py
    |   └── boosting.py
    │
    └── utils              <- utils files
        └── utils.py
```

### Setting
```
conda env create -f environment.yaml  # might be optional
conda activate amex
```

### Reference
+ [DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
+ [rounding-trick](https://www.kaggle.com/code/jiweiliu/amex-catboost-rounding-trick)
+ [time-features](https://www.kaggle.com/code/cdeotte/time-series-eda)
+ [pay-features](https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb)
+ [Ensemble model](https://www.sciencedirect.com/science/article/pii/S0957417421003407)
+ [feature-engineering](https://www.kaggle.com/code/susnato/amex-data-preprocesing-feature-engineering)
+ [statement-features](https://www.kaggle.com/code/romaupgini/statement-dates-to-use-or-not-to-use)
+ [seed-number](https://paperswithcode.com/paper/torch-manual-seed-3407-is-all-you-need-on-the)
+ [clean-data](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514)
--------
Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) & [microsoft recommenders](https://github.com/microsoft/recommenders/tree/main/recommenders).
