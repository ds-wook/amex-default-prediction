import logging

import hydra
from omegaconf import DictConfig

from data.dataset import load_test_dataset, load_train_dataset
from features.select import select_features
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = reduce_mem_usage(train_x)
    test_x = load_test_dataset(cfg)

    # calculate shap values
    features = select_features(train_x, train_y, test_x)

    logging.info(f"{features}")


if __name__ == "__main__":
    _main()
