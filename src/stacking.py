from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from xgboost import XGBClassifier

from models.infer import load_model


@hydra.main(config_path="../config/", config_name="stacking.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)

    lgbm_oofs1 = load_model(cfg, cfg.model.model1_oof)
    lgbm_oofs2 = load_model(cfg, cfg.model.model2_oof)
    lgbm_oofs3 = load_model(cfg, cfg.model.model3_oof)
    lgbm_oofs4 = load_model(cfg, cfg.model.model4_oof)

    target = train_labels["target"].to_numpy()
    oof_array = np.column_stack(
        [
            lgbm_oofs1.oof_preds,
            lgbm_oofs2.oof_preds,
            lgbm_oofs3.oof_preds,
            lgbm_oofs4.oof_preds,
        ]
    )

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    lgbm_preds4 = pd.read_csv(path / cfg.output.name / cfg.output.model4_preds)

    preds_array = np.column_stack(
        [
            lgbm_preds1.prediction.to_numpy(),
            lgbm_preds2.prediction.to_numpy(),
            lgbm_preds3.prediction.to_numpy(),
            lgbm_preds4.prediction.to_numpy(),
        ]
    )

    logit = XGBClassifier()
    logit.fit(oof_array, target)
    preds = logit.predict_proba(preds_array)[:, 1]
    submission["prediction"] = preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
