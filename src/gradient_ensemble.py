import logging
import warnings
from functools import partial
from pathlib import Path
from typing import List, NoReturn

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.optimize import fmin, minimize
from sklearn.model_selection import KFold

from evaluation.evaluate import amex_metric
from models.infer import load_model

warnings.filterwarnings("ignore")


class OptimizeAmex:
    def __init__(self) -> NoReturn:
        self.coef_ = 0

    def _amex(self, coef: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        amex_score = amex_metric(y, predictions)
        return -1.0 * amex_score

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NoReturn:
        partial_loss = partial(self._amex, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions


def get_score(
    weights: np.ndarray,
    train_idx: List[int],
    oofs: List[np.ndarray],
    preds: List[np.ndarray],
) -> float:
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]

    scores = amex_metric(preds[train_idx], blending)

    return -1 * scores


def get_best_weights(oofs: List[np.ndarray], preds: np.ndarray) -> float:
    weight_list = []
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])

    logging.info("Blending Start")
    kf = KFold(n_splits=5)
    for fold, (train_idx, _) in enumerate(kf.split(oofs[0]), 1):
        res = minimize(
            get_score,
            weights,
            args=(train_idx, oofs, preds),
            method="Nelder-Mead",
            tol=1e-6,
        )
        logging.info(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    mean_weight = np.insert(mean_weight, len(mean_weight), 1 - np.sum(mean_weight))
    logging.info(f"optimized weight: {mean_weight}\n")

    return mean_weight


@hydra.main(config_path="../config/", config_name="gradient.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)
    target = train_labels["target"]
    lgbm_oofs1 = load_model(cfg, cfg.model.model1_oof)
    lgbm_oofs2 = load_model(cfg, cfg.model.model2_oof)
    lgbm_oofs3 = load_model(cfg, cfg.model.model3_oof)
    lgbm_oofs4 = load_model(cfg, cfg.model.model4_oof)
    lgbm_oofs5 = load_model(cfg, cfg.model.model5_oof)
    lgbm_oofs6 = load_model(cfg, cfg.model.model6_oof)

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    lgbm_preds4 = pd.read_csv(path / cfg.output.name / cfg.output.model4_preds)
    lgbm_preds5 = pd.read_csv(path / cfg.output.name / cfg.output.model5_preds)
    lgbm_preds6 = pd.read_csv(path / cfg.output.name / cfg.output.model6_preds)

    oofs = [
        lgbm_oofs1.oof_preds,
        lgbm_oofs2.oof_preds,
        lgbm_oofs3.oof_preds,
        lgbm_oofs4.oof_preds,
        lgbm_oofs5.oof_preds,
        lgbm_oofs6.oof_preds,
    ]

    preds = [
        lgbm_preds1.prediction.to_numpy(),
        lgbm_preds2.prediction.to_numpy(),
        lgbm_preds3.prediction.to_numpy(),
        lgbm_preds4.prediction.to_numpy(),
        lgbm_preds5.prediction.to_numpy(),
        lgbm_preds6.prediction.to_numpy(),
    ]

    best_weights = get_best_weights(oofs, target.to_numpy())

    oof_preds = np.average(oofs, weights=best_weights, axis=0)
    print(f"OOF Score: {amex_metric(target.to_numpy(), oof_preds)}")
    blending_preds = np.average(preds, weights=best_weights, axis=0)
    submission["prediction"] = blending_preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
