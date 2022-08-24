from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from test_data.test_dataset import load_train_dataset
from test_features.test_select import test_selected_permutation_importances

from utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)

    # load train test dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = train_x.fillna(-127)
    train_x = train_x.replace((np.inf, -np.inf, np.nan), -127).reset_index(drop=True)

    pi_features = test_selected_permutation_importances(train_x, train_y, cfg)
    pi_features.to_csv(Path(get_original_cwd()) / "input/pi_features.csv", index=False)


if __name__ == "__main__":
    _main()
