import logging
import warnings
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from evaluation.evaluate import amex_metric
from models.infer import load_model

warnings.filterwarnings("ignore")


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
    for fold, (train_idx, valid_idx) in enumerate(kf.split(oofs[0])):
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


@hydra.main(config_path="../config/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)
    target = train_labels["target"]
    lgbm_oofs1 = load_model(cfg, cfg.model.model1_oof)
    lgbm_oofs2 = load_model(cfg, cfg.model.model2_oof)
    lgbm_oofs3 = load_model(cfg, cfg.model.model3_oof)

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)

    oofs = [lgbm_oofs1.oof_preds, lgbm_oofs2.oof_preds, lgbm_oofs3.oof_preds]

    preds = [
        lgbm_preds1.prediction.to_numpy(),
        lgbm_preds2.prediction.to_numpy(),
        lgbm_preds3.prediction.to_numpy(),
    ]

    best_weights = get_best_weights(oofs, target.to_numpy())

    oof_preds = np.average(oofs, weights=best_weights, axis=0)
    print(amex_metric(target.to_numpy(), lgbm_oofs1.oof_preds))
    print(amex_metric(target.to_numpy(), lgbm_oofs2.oof_preds))
    print(f"OOF Score: {amex_metric(target.to_numpy(), oof_preds)}")

    blending_preds = np.average(preds, weights=best_weights, axis=0)

    submission["prediction"] = blending_preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
