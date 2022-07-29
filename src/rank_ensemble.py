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


@hydra.main(config_path="../config/", config_name="rank.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd())
    submission = pd.read_csv(path / cfg.output.name / cfg.output.submission)
    submission["prediction"] = 0

    model1_preds = pd.read_csv(path / cfg.output.name / cfg.output.model1_preds)
    model2_preds = pd.read_csv(path / cfg.output.name / cfg.output.model2_preds)
    model3_preds = pd.read_csv(path / cfg.output.name / cfg.output.model3_preds)
    preds = [model1_preds, model2_preds, model3_preds]
    preds = [pred.sort_values(by="customer_ID") for pred in preds]

    for pred in tqdm(preds):
        pred["prediction"] = np.clip(pred["prediction"], 0, 1)

    for pred in tqdm(preds):
        submission["prediction"] += rankdata(pred["prediction"]) / pred.shape[0]

    submission["prediction"] /= len(preds)
    submission.to_csv(path / cfg.output.name / cfg.output.preds, index=False)


if __name__ == "__main__":
    _main()
