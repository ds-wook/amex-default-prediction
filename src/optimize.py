import warnings
from functools import partial
from pathlib import Path

import hydra
import optuna
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from optuna.trial import FrozenTrial

from evaluation.evaluate import amex_metric

warnings.filterwarnings("ignore")


def optimize_objective(
    trial: FrozenTrial,
    oof1: pd.Series,
    oof2: pd.Series,
    target: pd.Series,
    max_weights: float = 1.0,
) -> float:
    weights = trial.suggest_uniform("weights", 0, max_weights)
    blending = (1 - weights) * oof1 + weights * oof2

    return amex_metric(target, blending)


@hydra.main(config_path="../config/", config_name="optimize", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)
    target = train_labels["target"]

    lgbm_oof = pd.read_csv(path / cfg.model.path / cfg.model.lgbm_oof)
    xgb_oof = pd.read_csv(path / cfg.model.path / cfg.model.xgb_oof)
    tabnet_oof = pd.read_csv(path / cfg.model.path / cfg.model.tabnet_oof)

    lgbm_preds = pd.read_csv(path / cfg.output.name / cfg.output.lgbm_preds)
    xgb_preds = pd.read_csv(path / cfg.output.name / cfg.output.xgb_preds)
    tabnet_preds = pd.read_csv(path / cfg.output.name / cfg.output.tabnet_preds)

    params = {"weights": 0.03221084221328546}
    oof_preds = (
        lgbm_oof.prediction * (1 - params["weights"])
        + xgb_oof.prediction * params["weights"]
    )
    blending_preds = (
        lgbm_preds.prediction * (1 - params["weights"])
        + xgb_preds.prediction * params["weights"]
    )

    objective = partial(
        optimize_objective,
        oof1=oof_preds,
        oof2=tabnet_oof.prediction,
        target=target,
        max_weights=0.001,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    params = study.best_params
    oof_preds = (
        oof_preds * (1 - params["weights"]) + tabnet_oof.prediction * params["weights"]
    )
    blending_preds = (
        blending_preds * (1 - params["weights"])
        + tabnet_preds.prediction * params["weights"]
    )
    print(f"OOF Score: {amex_metric(target.to_numpy(), oof_preds)}")

    train_labels["prediction"] = oof_preds
    train_labels.to_csv(path / cfg.model.path / cfg.output.oof, index=False)
    submission["prediction"] = blending_preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
