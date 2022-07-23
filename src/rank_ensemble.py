import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from scipy.stats import rankdata
from tqdm import tqdm

warnings.filterwarnings("ignore")


@hydra.main(config_path="../config/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    submission["prediction"] = 0

    lgbm_preds1 = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    lgbm_preds2 = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    lgbm_preds3 = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    lgbm_preds4 = pd.read_csv(path / cfg.output.name / cfg.output.model4_preds)
    overfitting = pd.read_csv(path / cfg.output.name / cfg.output.overfitting1)

    preds = [lgbm_preds1, lgbm_preds2, lgbm_preds3, lgbm_preds4, overfitting]
    preds = [pred.sort_values(by="customer_ID") for pred in preds]

    for pred in tqdm(preds):
        pred["prediction"] = np.clip(pred["prediction"], 0, 1)

    for pred in tqdm(preds):
        submission["prediction"] += rankdata(pred["prediction"]) / pred.shape[0]

    submission["prediction"] /= len(preds)
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
