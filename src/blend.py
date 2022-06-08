from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.name
    tabnet_preds = pd.read_csv(path / cfg.output.tabnet_preds)
    nn_preds = pd.read_csv(path / cfg.output.nn_preds)
    lgbm_preds = pd.read_csv(path / cfg.output.lgbm_preds)
    xgb_preds = pd.read_csv(path / cfg.output.xgb_preds)
    cb_preds = pd.read_csv(path / cfg.output.cb_preds)
    lb_preds = pd.read_csv(path / cfg.output.lb_preds)

    ensemble_preds = pd.merge(nn_preds, tabnet_preds, on="customer_ID")
    ensemble_preds.rename(
        columns={"prediction_x": "nn_preds", "prediction_y": "gru_preds"},
        inplace=True,
    )
    ensemble_preds["prediction"] = (
        ensemble_preds["nn_preds"] * 0.6 + ensemble_preds["gru_preds"] * 0.4
    )

    tree_ensemble_preds = pd.merge(cb_preds, xgb_preds, on="customer_ID")
    tree_ensemble_preds.rename(
        columns={"prediction_x": "cb_preds", "prediction_y": "xgb_preds"},
        inplace=True,
    )
    tree_ensemble_preds["prediction"] = (
        tree_ensemble_preds["cb_preds"] * 0.3 + tree_ensemble_preds["xgb_preds"] * 0.7
    )

    preds = nn_preds.copy()
    preds["prediction"] = (
        ensemble_preds["prediction"] * 0.2
        + tree_ensemble_preds["prediction"] * 0.2
        + lgbm_preds["prediction"] * 0.6
    )
    preds["prediction"] = preds["prediction"] * 0.5 + lb_preds["prediction"] * 0.5
    preds.to_csv(path / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
