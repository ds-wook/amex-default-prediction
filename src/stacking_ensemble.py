from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from models.infer import load_model
from models.stacking import StackingDataLoder


@hydra.main(config_path="../config/", config_name="stacking.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.dataset.path

    oof_pred1 = load_model(cfg, cfg.input.oof_pred1)
    oof_pred2 = load_model(cfg, cfg.input.oof_pred2)
    oof_pred3 = load_model(cfg, cfg.input.oof_pred3)
    oof_pred4 = load_model(cfg, cfg.input.oof_pred4)

    oof_array = np.column_stack(
        [
            oof_pred1.oof_preds,
            oof_pred2.oof_preds,
            oof_pred3.oof_preds,
            oof_pred4.oof_preds,
        ]
    )

    path = Path(get_original_cwd()) / cfg.output.path
    preds1 = pd.read_csv(path / cfg.output.preds1)
    preds2 = pd.read_csv(path / cfg.output.preds2)
    preds3 = pd.read_csv(path / cfg.output.preds3)
    preds4 = pd.read_csv(path / cfg.output.preds4)

    preds_array = np.column_stack(
        [
            preds1.prediction.to_numpy(),
            preds2.prediction.to_numpy(),
            preds3.prediction.to_numpy(),
            preds4.prediction.to_numpy(),
        ]
    )
    # fmt: on
    stacking_dataloder = StackingDataLoder(cfg, oof_array, preds_array)
    meta_train = stacking_dataloder.make_train_dataset()
    meta_test = stacking_dataloder.make_test_dataset()

    meta_train.to_csv(path / cfg.output.train, index=False)
    meta_test.to_csv(path / cfg.output.test, index=False)


if __name__ == "__main__":
    _main()
