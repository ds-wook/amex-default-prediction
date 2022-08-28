from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from evaluation.evaluate import amex_metric, xgb_amex_metric
from models.infer import load_model


# DEFINE CUSTOM EVAL LOSS FUNCTION
def weighted_logloss(
    preds: np.ndarray,
    dtrain: xgb.DMatrix,
    mult_no4prec: float = 5.0,
    max_weights: float = 2.0,
) -> Tuple[float, float]:
    labels = dtrain.get_label()

    preds = 1.0 / (1.0 + np.exp(-preds))

    # top 4%
    labels_mat = np.transpose(np.array([np.arange(len(labels)), labels, preds]))
    pos_ord = labels_mat[:, 2].argsort()[::-1]
    labels_mat = labels_mat[pos_ord]
    weights_4perc = np.where(labels_mat[:, 1] == 0, 20, 1)
    top4 = np.cumsum(weights_4perc) <= int(0.04 * np.sum(weights_4perc))
    top4 = top4[labels_mat[:, 0].argsort()]

    weights = (
        1
        + np.exp(-mult_no4prec * np.linspace(max_weights - 1, 0, len(top4)))[
            labels_mat[:, 0].argsort()
        ]
    )

    # Set to one weights of positive labels in top 4perc
    weights[top4 & (labels == 1.0)] = 1.0
    # Set to one weights of negative labels
    weights[(labels == 0.0)] = 1.0

    grad = preds * (1 + weights * labels - labels) - (weights * labels)
    hess = preds * (1 - preds) * (1 + weights * labels - labels)
    return grad, hess


@hydra.main(config_path="../config/", config_name="stacking", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)

    lgbm_oofs1 = load_model(cfg, cfg.model.model1_oof)
    lgbm_oofs2 = load_model(cfg, cfg.model.model2_oof)
    lgbm_oofs3 = load_model(cfg, cfg.model.model3_oof)
    lgbm_oofs4 = load_model(cfg, cfg.model.model4_oof)
    lgbm_oofs5 = load_model(cfg, cfg.model.model5_oof)
    lgbm_oofs6 = load_model(cfg, cfg.model.model6_oof)
    lgbm_oofs7 = load_model(cfg, cfg.model.model7_oof)
    lgbm_oofs8 = load_model(cfg, cfg.model.model8_oof)
    lgbm_oofs9 = load_model(cfg, cfg.model.model9_oof)
    lgbm_oofs10 = load_model(cfg, cfg.model.model10_oof)
    lgbm_oofs11 = load_model(cfg, cfg.model.model11_oof)
    lgbm_oofs12 = load_model(cfg, cfg.model.model12_oof)
    lgbm_oofs13 = load_model(cfg, cfg.model.model13_oof)
    lgbm_oofs14 = load_model(cfg, cfg.model.model14_oof)
    lgbm_oofs15 = load_model(cfg, cfg.model.model15_oof)
    target = train_labels["target"]

    oof_array = np.column_stack(
        [
            lgbm_oofs1.oof_preds,
            lgbm_oofs2.oof_preds,
            lgbm_oofs3.oof_preds,
            lgbm_oofs4.oof_preds,
            lgbm_oofs5.oof_preds,
            lgbm_oofs6.oof_preds,
            lgbm_oofs7.oof_preds,
            lgbm_oofs8.oof_preds,
            lgbm_oofs9.oof_preds,
            lgbm_oofs10.oof_preds,
            lgbm_oofs11.oof_preds,
            lgbm_oofs12.oof_preds,
            lgbm_oofs13.oof_preds,
            lgbm_oofs14.oof_preds,
            lgbm_oofs15.oof_preds,
        ]
    )

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    lgbm_preds4 = pd.read_csv(path / cfg.output.name / cfg.output.model4_preds)
    lgbm_preds5 = pd.read_csv(path / cfg.output.name / cfg.output.model5_preds)
    lgbm_preds6 = pd.read_csv(path / cfg.output.name / cfg.output.model6_preds)
    lgbm_preds7 = pd.read_csv(path / cfg.output.name / cfg.output.model7_preds)
    lgbm_preds8 = pd.read_csv(path / cfg.output.name / cfg.output.model8_preds)
    lgbm_preds9 = pd.read_csv(path / cfg.output.name / cfg.output.model9_preds)
    lgbm_preds10 = pd.read_csv(path / cfg.output.name / cfg.output.model10_preds)
    lgbm_preds11 = pd.read_csv(path / cfg.output.name / cfg.output.model11_preds)
    lgbm_preds12 = pd.read_csv(path / cfg.output.name / cfg.output.model12_preds)
    lgbm_preds13 = pd.read_csv(path / cfg.output.name / cfg.output.model13_preds)
    lgbm_preds14 = pd.read_csv(path / cfg.output.name / cfg.output.model14_preds)
    lgbm_preds15 = pd.read_csv(path / cfg.output.name / cfg.output.model15_preds)

    preds_array = np.column_stack(
        [
            lgbm_preds1.prediction.to_numpy(),
            lgbm_preds2.prediction.to_numpy(),
            lgbm_preds3.prediction.to_numpy(),
            lgbm_preds4.prediction.to_numpy(),
            lgbm_preds5.prediction.to_numpy(),
            lgbm_preds6.prediction.to_numpy(),
            lgbm_preds7.prediction.to_numpy(),
            lgbm_preds8.prediction.to_numpy(),
            lgbm_preds9.prediction.to_numpy(),
            lgbm_preds10.prediction.to_numpy(),
            lgbm_preds11.prediction.to_numpy(),
            lgbm_preds12.prediction.to_numpy(),
            lgbm_preds13.prediction.to_numpy(),
            lgbm_preds14.prediction.to_numpy(),
            lgbm_preds15.prediction.to_numpy(),
        ]
    )

    oof_df = pd.DataFrame(
        oof_array, columns=[f"preds_{i}" for i in range(1, oof_array.shape[1] + 1)]
    )

    preds_df = pd.DataFrame(
        preds_array, columns=[f"preds_{i}" for i in range(1, preds_array.shape[1] + 1)]
    )

    # save predictions
    str_kf = StratifiedKFold(
        n_splits=cfg.model.fold, shuffle=True, random_state=cfg.dataset.seed
    )
    splits = str_kf.split(oof_df, target)
    oof_preds = np.zeros(len(oof_df))
    preds_proba = np.zeros(len(preds_df))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # split train and validation data
        X_train, y_train = oof_df.iloc[train_idx], target.iloc[train_idx]
        X_valid, y_valid = oof_df.iloc[valid_idx], target.iloc[valid_idx]

        # model
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dict(cfg.model.params),
            dtrain=dtrain,
            evals=watchlist,
            feval=xgb_amex_metric,
            obj=weighted_logloss,
            maximize=True,
            num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose_eval=cfg.model.verbose,
        )

        # validation
        oof_preds[valid_idx] = model.predict(xgb.DMatrix(X_valid))
        preds_proba += model.predict(xgb.DMatrix(preds_df)) / 10

        # score
        score = amex_metric(y_valid, oof_preds[valid_idx])
        print(f"Fold {fold}: {score}")

        del X_train, X_valid, y_train, y_valid, model

    oof_score = amex_metric(target, oof_preds)
    print(f"OOF Score: {oof_score}")
    oof_df = pd.DataFrame(
        {
            "customer_ID": train_labels["customer_ID"],
            "target": train_labels[cfg.dataset.target],
            "prediction": oof_preds,
        }
    )
    oof_df.to_csv(path / cfg.output.name / cfg.output.oof, index=False)
    submission["prediction"] = preds_proba
    submission.to_csv(path / cfg.model.path / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
