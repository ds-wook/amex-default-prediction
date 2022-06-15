import logging

import hydra
from omegaconf import DictConfig

from data.dataset import load_test_dataset, load_train_dataset
from features.build import create_categorical_test, create_categorical_train
from features.select import select_features


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
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

    logging.info(f"{features}")


if __name__ == "__main__":
    _main()
