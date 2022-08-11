import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from data.dataset import load_train_dataset
from features.select import get_score_correlation, score_feature_selection
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)

    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)

    train_x, _, train_y, _ = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
    )

    # actual_imp_df = pd.read_csv(Path(get_original_cwd()) / "input/actual_imp.csv")
    # null_imp_df = pd.read_csv(Path(get_original_cwd()) / "input/null_imp.csv")

    # _, correlation_scores = get_score_correlation(
    #     actual_imp_df, null_imp_df
    # )
    # split_feats = [_f for _f, _score, _ in correlation_scores if _score > 20]
    # split_cat_feats = [
    #     _f
    #     for _f, _score, _ in correlation_scores
    #     if (_score > 20) and (_f in cfg.dataset.cat_features)
    # ]

    shap_importance_df = pd.read_csv(
        Path(get_original_cwd()) / "input/shap_importance.csv"
    )

    shap_importance = shap_importance_df[shap_importance_df["shap_importance"] > 0.001]
    not_importance = shap_importance_df[shap_importance_df["shap_importance"] < 0.001]
    not_importance_features = not_importance["column_name"].values.tolist()
    shap_features = shap_importance["column_name"].values.tolist()
    sdist_features = [
        "LT",
        "SDist_last",
        "SDist_first",
        "SDist_mean",
        "SDist_max",
        "SDist_min",
    ]
    last_features = [
        col for col in not_importance_features if "last" in col and "SDist" not in col
    ]
    selected_features = shap_features + sdist_features + last_features

    selected_cat_features = [
        col for col in selected_features if col in cfg.dataset.cat_features
    ]
    # selected_cat_features += split_cat_feats

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
