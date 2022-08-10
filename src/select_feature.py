import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from data.dataset import load_train_dataset
from features.select import score_feature_selection
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)

    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)

    train_x, _, train_y, _ = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
    )

    shap_importance_df = pd.read_csv(
        Path(get_original_cwd()) / "input/test_shap_importance.csv"
    )
    shap_importance_df2 = pd.read_csv(
        Path(get_original_cwd()) / "input/test_shap_importance2.csv"
    )

    shap_importance_df = shap_importance_df[shap_importance_df["shap_importance"] != 0]
    shap_scores = shap_importance_df["column_name"].values.tolist()

    shap_importance_df2 = shap_importance_df2[
        shap_importance_df2["shap_importance"] == 0
    ]
    shap_scores2 = shap_importance_df2["column_name"].values.tolist()

    selected_features = shap_scores + shap_scores2
    selected_cat_features = [
        col for col in selected_features if col in cfg.dataset.cat_features
    ]

    logging.info(f"Select {len(selected_features)}")
    split_results = score_feature_selection(
        df=train_x,
        train_features=selected_features,
        cat_features=selected_cat_features,
        target=train_y,
    )
    logging.info("\t SHAP : %.6f +/- %.6f" % (split_results[0], split_results[1]))

    path = Path(get_original_cwd()) / cfg.features.path

    update_features = OmegaConf.load(path / cfg.features.name)
    update_features.selected_features = selected_features
    update_features.cat_features = selected_cat_features
    OmegaConf.save(update_features, path / cfg.features.name)


if __name__ == "__main__":
    _main()
