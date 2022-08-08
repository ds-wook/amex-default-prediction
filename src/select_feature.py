from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_test_dataset, load_train_dataset
from features.select import select_features


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)
    test_x = load_test_dataset(cfg)
    # calculate shap values
    selected_features, select_cat_features, importance_df = select_features(
        train_x, train_y, test_x, cfg.model.params.seed
    )

    # save selected features
    path = Path(get_original_cwd()) / cfg.features.path
    importance_df.to_csv(path / "shap_importance.csv", index=False)
    update_features = OmegaConf.load(path / cfg.features.name)
    update_features.selected_features = selected_features
    update_features.cat_features = select_cat_features
    OmegaConf.save(update_features, path / cfg.features.name)


if __name__ == "__main__":
    _main()
