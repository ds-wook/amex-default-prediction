import logging
import warnings
from functools import partial
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from evaluation.evaluate import amex_metric
from models.infer import load_model

warnings.filterwarnings("ignore")


def get_score(
    weights: np.ndarray,
    train_idx: List[int],
    oofs: List[np.ndarray],
    target: np.ndarray,
    score: str,
) -> float:
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]

    scores = (
        log_loss(target[train_idx], blending)
        if score == "log_loss"
        else amex_metric(target[train_idx], blending)
    )
    return scores


def get_best_weights(oofs: List[np.ndarray], target: np.ndarray, score: str) -> float:
    weight_list = []
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])

    logging.info("Blending Start")
    kf = KFold(n_splits=5)
    for fold, (train_idx, _) in enumerate(kf.split(oofs[0]), 1):
        res = minimize(
            partial(get_score, score=score),
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


@hydra.main(config_path="../config/", config_name="gradient.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)
    target = train_labels["target"]

    tabnet_oof = pd.read_csv(path / cfg.model.path / cfg.model.tabnet_oof)
    lgbm1_oof = pd.read_csv(path / cfg.model.path / cfg.model.lgbm1_oof)
    lgbm2_oof = pd.read_csv(path / cfg.model.path / cfg.model.lgbm2_oof)
    cb1_oof = pd.read_csv(path / cfg.model.path / cfg.model.cb1_oof)
    cb2_oof = pd.read_csv(path / cfg.model.path / cfg.model.cb2_oof)

    tabnet_preds = pd.read_csv(path / cfg.output.name / cfg.output.tabnet_preds)
    lgbm1_preds = pd.read_csv(path / cfg.output.name / cfg.output.lgbm1_preds)
    lgbm2_preds = pd.read_csv(path / cfg.output.name / cfg.output.lgbm2_preds)
    cb1_preds = pd.read_csv(path / cfg.output.name / cfg.output.cb1_preds)
    cb2_preds = pd.read_csv(path / cfg.output.name / cfg.output.cb2_preds)

    oofs = [
        lgbm1_oof.prediction.to_numpy(),
        lgbm2_oof.prediction.to_numpy(),
        cb1_oof.prediction.to_numpy(),
        # cb2_oof.prediction.to_numpy(),
    ]

    preds = [
        lgbm1_preds.prediction.to_numpy(),
        lgbm2_preds.prediction.to_numpy(),
        cb1_preds.prediction.to_numpy(),
        # cb2_preds.prediction.to_numpy(),
    ]

    best_weights = get_best_weights(oofs, target.to_numpy(), cfg.score.name)

    oof_preds = np.average(oofs, weights=best_weights, axis=0)
    blending_preds = np.average(preds, weights=best_weights, axis=0)

    print(f"OOF Score: {amex_metric(target.to_numpy(), oof_preds)}")

    train_labels["prediction"] = oof_preds
    train_labels.to_csv(path / cfg.model.path / cfg.output.oof, index=False)
    submission["prediction"] = blending_preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
