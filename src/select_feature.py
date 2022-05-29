import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from features.select import select_features


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)

    features = select_features(train_x, train_y)

    print(features)


if __name__ == "__main__":
    _main()
