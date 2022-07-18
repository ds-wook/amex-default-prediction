from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from evaluation.evaluate import amex_metric
from models.boosting import XGBoostTrainer
from models.infer import inference, load_model


@hydra.main(config_path="../config/", config_name="stacking.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    train_labels = pd.read_csv(path / cfg.input.name / cfg.input.train_labels)

    lgbm_oofs1 = load_model(cfg, cfg.model.model1_oof)
    lgbm_oofs2 = load_model(cfg, cfg.model.model2_oof)
    lgbm_oofs3 = load_model(cfg, cfg.model.model3_oof)
    lgbm_oofs4 = load_model(cfg, cfg.model.model4_oof)
    lgbm_oofs5 = load_model(cfg, cfg.model.model5_oof)
    lgbm_oofs6 = load_model(cfg, cfg.model.model6_oof)
    lgbm_oofs7 = load_model(cfg, cfg.model.model7_oof)
    lgbm_oofs8 = load_model(cfg, cfg.model.model8_oof)
    lgbm_oofs9 = load_model(cfg, cfg.model.model9_oof)
    lgbm_oofs10 = load_model(cfg, cfg.model.model10_oof)

    target = train_labels["target"]
    oof_array = np.column_stack(
        [
            lgbm_oofs1.oof_preds,
            lgbm_oofs2.oof_preds,
            lgbm_oofs3.oof_preds,
            lgbm_oofs4.oof_preds,
            lgbm_oofs5.oof_preds,
            lgbm_oofs6.oof_preds,
            lgbm_oofs7.oof_preds,
            lgbm_oofs8.oof_preds,
            lgbm_oofs9.oof_preds,
            lgbm_oofs10.oof_preds,
        ]
    )

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    lgbm_preds4 = pd.read_csv(path / cfg.output.name / cfg.output.model4_preds)
    lgbm_preds5 = pd.read_csv(path / cfg.output.name / cfg.output.model5_preds)
    lgbm_preds6 = pd.read_csv(path / cfg.output.name / cfg.output.model6_preds)
    lgbm_preds7 = pd.read_csv(path / cfg.output.name / cfg.output.model7_preds)
    lgbm_preds8 = pd.read_csv(path / cfg.output.name / cfg.output.model8_preds)
    lgbm_preds9 = pd.read_csv(path / cfg.output.name / cfg.output.model9_preds)
    lgbm_preds10 = pd.read_csv(path / cfg.output.name / cfg.output.model10_preds)

    preds_array = np.column_stack(
        [
            lgbm_preds1.prediction.to_numpy(),
            lgbm_preds2.prediction.to_numpy(),
            lgbm_preds3.prediction.to_numpy(),
            lgbm_preds4.prediction.to_numpy(),
            lgbm_preds5.prediction.to_numpy(),
            lgbm_preds6.prediction.to_numpy(),
            lgbm_preds7.prediction.to_numpy(),
            lgbm_preds8.prediction.to_numpy(),
            lgbm_preds9.prediction.to_numpy(),
            lgbm_preds10.prediction.to_numpy(),
        ]
    )

    oof_df = pd.DataFrame(oof_array, columns=[f"preds_{i}" for i in range(1, 11)])
    preds_df = pd.DataFrame(preds_array, columns=[f"preds_{i}" for i in range(1, 11)])
    xgb_trainer = XGBoostTrainer(config=cfg, metric=amex_metric)
    xgb_trainer.train(oof_df, target)
    preds = inference(xgb_trainer, preds_df)

    submission["prediction"] = preds
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
