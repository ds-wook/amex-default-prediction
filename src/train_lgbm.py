import hydra
from omegaconf import DictConfig

from data.dataset import load_dataset
from models.boosting import LightGBMTrainer
from utils.utils import amex_metric


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    train_x, train_y, test_x = load_dataset(cfg)

    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
