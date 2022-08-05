import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from models.boosting import XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    lgb_trainer = XGBoostTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
