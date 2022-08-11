from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from data.dataset import load_test_dataset, load_train_dataset
from features.select import get_shap_importances, get_score_correlation
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)
    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)

    actual_imp_df = pd.read_csv(Path(get_original_cwd()) / "input/actual_imp.csv")
    null_imp_df = pd.read_csv(Path(get_original_cwd()) / "input/null_imp.csv")

    correlation_scores_df, correlation_scores = get_score_correlation(
        actual_imp_df, null_imp_df
    )
    # shap_importance_df = pd.read_csv(
    #     Path(get_original_cwd()) / "input/shap_importance.csv"
    # )
    # shap_importance_df2 = pd.read_csv(
    #     Path(get_original_cwd()) / "input/shap_importance2.csv"
    # )
    # shap_importance_df = shap_importance_df[shap_importance_df["shap_importance"] != 0]
    # shap_importance_df2 = shap_importance_df2[
    #     shap_importance_df2["shap_importance"] != 0
    # ]
    # shap_columns = (
    #     shap_importance_df["column_name"].values.tolist()
    #     + shap_importance_df2["column_name"].values.tolist()
    # )
    # cat_features = [col for col in shap_columns if col in cfg.dataset.cat_features]
    # train_x = train_x[shap_columns]
    train_x, _, train_y, _ = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
    )
    split_feats = [_f for _f, _score, _ in correlation_scores if _score < 20]
    split_cat_feats = [
        _f
        for _f, _score, _ in correlation_scores
        if (_score < 20) and (_f in cfg.dataset.cat_features)
    ]
    train_x = train_x[split_feats]
    test_sample = load_test_dataset(cfg)
    test_sample = test_sample[split_feats]
    shap_importance_df = get_shap_importances(
        train_x, train_y, test_sample, split_cat_feats
    )
    shap_importance_df.to_csv(
        Path(get_original_cwd()) / "input/null_shap_importance.csv", index=False
    )


if __name__ == "__main__":
    _main()
