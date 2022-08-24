from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.metrics import Metric


class AmexMetric(Metric):
    def __init__(self):
        self._name = "amex-metric"
        self._maximize = True

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        score = amex_metric(y_true, y_score[:, 1])
        return score


class CatBoostEvalMetricAmex:
    def get_final_error(self, error: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return error

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(
        self, approxes: np.ndarray, target: np.ndarray, weight: np.ndarray
    ) -> Tuple[float, int]:
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        return amex_metric(np.array(target), np.array(preds)), 0


def amex_metric(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]

    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def amex_metric_numpy(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    target = y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


def lgb_amex_metric(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[str, float, bool]:
    """The competition metric with lightgbm's calling convention"""
    y_true = y_true.get_label()
    return "amex", amex_metric(y_true, y_pred), True


def xgb_amex_metric(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """The competition metric with xgboost's calling convention"""
    y_true = dtrain.get_label()
    return "amex", amex_metric(y_true, y_pred)
