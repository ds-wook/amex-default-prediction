import logging

import hydra
from omegaconf import DictConfig

from features.select import select_features
from utils.utils import timer


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    # timer
    with timer("Select features"):
        # calculate shap values
        features = select_features(cfg)

        logging.info(f"{features}")


if __name__ == "__main__":
    _main()
