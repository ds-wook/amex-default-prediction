from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.name
    nn_preds = pd.read_csv(path / cfg.output.nn_preds)
    lgbm_preds = pd.read_csv(path / cfg.output.lgbm_preds)
    cb_preds = pd.read_csv(path / cfg.output.cb_preds)

    preds = nn_preds.copy()
    ensemble_perds = (
        + lgbm_preds["prediction"] * 0.5
        + nn_preds["prediction"] * 0.5
    )
    preds["prediction"] = ensemble_perds
    preds.to_csv(path / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()