import collections
import pickle
from typing import NoReturn

import lightgbm as lgb

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


class TestDartEarlyStopping:
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
