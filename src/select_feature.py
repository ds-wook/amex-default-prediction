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
    train_x = train_x.drop(columns=[*cfg.dataset.cat_features])
    test_x = load_test_dataset(cfg)
    test_x = test_x.drop(columns=[*cfg.dataset.cat_features])

    # calculate shap values
    features = select_features(
        train_x.iloc[:10000], train_y.iloc[:10000], test_x.iloc[:10000]
    )
    features.extend([*cfg.dataset.cat_features])
    # save selected features
    path = Path(get_original_cwd()) / cfg.features.path
    update_features = OmegaConf.load(path / cfg.features.name)
    update_features.selected_features = features
    OmegaConf.save(update_features, path / cfg.features.name)


if __name__ == "__main__":
    _main()
