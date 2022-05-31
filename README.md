## amex-default-prediction
Predict if a customer will default in the future

### Benchmark
|Model|CV|Public LB|
|-----|--|------|
|LightGBM(10-stratified kfold - gbdt)|0.789|0.791|
|LightGBM(5-stratified kfold - dart)|-|-|
|CatBoost(10-stratified categorical kfold)|0.789|0.793|
|LightGBM(10-stratified categorical kfold - gbdt)|0.790|0.792|
|**LightGBM(5-stratified categorical kfold - dart)**|**0.792**|**0.794**|

### Paper
+ [DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
