python src/train_lgbm.py 
\ model.path=res/models/10fold_lightgbm_dart_trick.pkl

python src/predict.py 
\ model.lightgbm=res/models/10fold_lightgbm_dart_trick.pkl 
\ output.name=10fold_lightgbm_dart_trick.csv
