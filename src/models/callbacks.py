import collections
import pickle
from typing import NoReturn, Tuple

import lightgbm as lgb
import numpy as np

# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "CallbackEnv",
    [
        "model",
        "params",
        "iteration",
        "begin_iteration",
        "end_iteration",
        "evaluation_result_list",
    ],
)


class DartEarlyStopping:
    def __init__(self, data_name: str, monitor_metric: str, stopping_round: int):
        self.data_name = data_name
        self.monitor_metric = monitor_metric
        self.stopping_round = stopping_round
        self.best_score = None
        self.best_model = None
        self.best_score_list = []
        self.best_iter = 0

    def _is_higher_score(self, metric_score: float, is_higher_better: float) -> bool:
        if self.best_score is None:
            return True
        return (
            (self.best_score < metric_score)
            if is_higher_better
            else (self.best_score > metric_score)
        )

    def _deepcopy(self, x: CallbackEnv) -> CallbackEnv:
        return pickle.loads(pickle.dumps(x))

    def __call__(self, env: CallbackEnv) -> NoReturn:
        evals = env.evaluation_result_list
        for data, metric, score, is_higher_better in evals:
            if data != self.data_name or metric != self.monitor_metric:
                continue
            if not self._is_higher_score(score, is_higher_better):
                if env.iteration - self.best_iter > self.stopping_round:
                    eval_result_str = "\t".join(
                        [
                            lgb.callback._format_eval_result(x)
                            for x in self.best_score_list
                        ]
                    )
                    lgb.basic._log_info(
                        f"Early stopping, best iteration is:\n[{self.best_iter+1}]\t{eval_result_str}"
                    )
                    lgb.basic._log_info(
                        'You can get best model by "DartEarlyStopping.best_model"'
                    )
                    raise lgb.callback.EarlyStopException(
                        self.best_iter, self.best_score_list
                    )
                return

            self.best_model = self._deepcopy(env.model)
            self.best_iter = env.iteration
            self.best_score_list = evals
            self.best_score = score
            return
        raise ValueError("monitoring metric not found")


def weighted_logloss(
    preds: np.ndarray, dtrain: lgb.Dataset, mult_no4prec: float, max_weights: float
) -> Tuple[float, float]:
    """
    weighted logloss for dart
    Args:
        preds: prediction
        dtrain: lgb.Dataset
        mult_no4prec: weight for no4prec
        max_weights: max weight for no4prec
    Returns:
        gradient, hessian
    """
    eps = 1e-16
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
    weights[
        top4 & (labels == 1.0)
    ] = 1.0  # Set to one weights of positive labels in top 4%
    weights[(labels == 0.0)] = 1.0  # Set to one weights of negative labels

    grad = (preds - labels) * weights
    hess = np.maximum(preds * (1.0 - preds) * weights, eps)
    return grad, hess
