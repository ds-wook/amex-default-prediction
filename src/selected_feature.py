from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_train_dataset
from features.select import selected_permutation_importances
from utils.utils import seed_everything


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)

    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = train_x.fillna(-127)
    train_x = train_x.replace((np.inf, -np.inf, np.nan), -127).reset_index(drop=True)

    pi_features = selected_permutation_importances(cfg, train_x, train_y)
    pi_features.to_csv(Path(get_original_cwd()) / "input/pi_feature.csv", index=False)
    selected_features = pi_features[pi_features["weight"] >= 0.0]["feature"].to_list()
    selected_cat_features = [
        col for col in selected_features if col in cfg.dataset.cat_features
    ]

    path = Path(get_original_cwd()) / cfg.features.path

    update_features = OmegaConf.load(path / cfg.features.name)
    update_features.selected_features = selected_features
    update_features.cat_features = selected_cat_features
    OmegaConf.save(update_features, path / cfg.features.name)


if __name__ == "__main__":
    _main()
