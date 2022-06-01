import logging
from typing import List

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def get_score(
    weights: np.ndarray, train_idx: List[int], oofs: List[np.ndarray], preds: np.ndarray
) -> float:
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]

    scores = mean_absolute_error(preds[train_idx], blending)

    return scores


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
