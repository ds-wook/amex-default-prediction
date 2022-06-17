from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_test_dataset, load_train_dataset
from features.build import create_categorical_test, create_categorical_train
from features.select import select_features


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    train_x.fillna(-127, inplace=True)
    test_x = load_test_dataset(cfg, 9)
    test_x = create_categorical_test(test_x, cfg)
    test_x.fillna(-127, inplace=True)
    # calculate shap values
    features = select_features(train_x, train_y, test_x)

    # save selected features
    path = Path(get_original_cwd()) / cfg.features.path
    update_features = OmegaConf.load(path / cfg.features.name)
    update_features.selected_features = features
    OmegaConf.save(update_features, path / cfg.features.name)


if __name__ == "__main__":
    _main()
