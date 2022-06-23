from typing import Tuple

import numpy as np
from pytorch_tabnet.metrics import Metric
from xgboost import DMatrix


class AmexMetric(Metric):
    def __init__(self):
        self._name = "amex-metric"
        self._maximize = True

    def __call__(self, y_true, y_score):
        score = amex_metric(y_true, y_score[:, 1])
        return score


class CatBoostEvalMetricAmex:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        return amex_metric(np.array(target), np.array(preds)), 0


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def lgb_amex_metric(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[str, float, bool]:
    """The competition metric with lightgbm's calling convention"""
    y_true = y_true.get_label()
    return "amex", amex_metric(y_true, y_pred), True


def xgb_amex_metric(y_pred: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    """The competition metric with xgboost's calling convention"""
    y_true = dtrain.get_label()
    return "amex", amex_metric(y_true, y_pred)
