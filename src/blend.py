from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.name
    gru_preds = pd.read_csv(path / cfg.output.gru_preds)
    nn_preds = pd.read_csv(path / cfg.output.nn_preds)
    lgbm_preds = pd.read_csv(path / cfg.output.lgbm_preds)
    cb_preds = pd.read_csv(path / cfg.output.cb_preds)

    ensemble_preds = pd.merge(lgbm_preds, gru_preds, on="customer_ID")
    ensemble_preds.rename(
        columns={"prediction_x": "lgbm_preds", "prediction_y": "gru_preds"},
        inplace=True,
    )
    ensemble_preds["prediction"] = (
        ensemble_preds["lgbm_preds"] * 0.9 + ensemble_preds["gru_preds"] * 0.1
    )

    preds = nn_preds.copy()
    preds["prediction"] = (
        ensemble_preds["prediction"] * 0.6
        + cb_preds["prediction"] * 0.2
        + nn_preds["prediction"] * 0.2
    )
    preds.to_csv(path / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
