from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from data.dataset import load_test_dataset, load_train_dataset
from features.select import get_score_correlation, get_shap_importances
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)
    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)

    shap_importance_df = pd.read_csv(
        Path(get_original_cwd()) / "input/first_shap_importance.csv"
    )

    shap_importance_df = shap_importance_df[shap_importance_df["shap_importance"] == 0]

    shap_columns = shap_importance_df["column_name"].values.tolist()
    cat_features = [col for col in shap_columns if col in cfg.dataset.cat_features]
    train_x = train_x[shap_columns]
    train_x, _, train_y, _ = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
    )

    test_sample = load_test_dataset(cfg)
    test_sample = test_sample[shap_columns]
    shap_importance_df = get_shap_importances(
        train_x, train_y, test_sample, cat_features
    )
    shap_importance_df.to_csv(
        Path(get_original_cwd()) / "input/first_shap_importance2.csv", index=False
    )


if __name__ == "__main__":
    _main()
