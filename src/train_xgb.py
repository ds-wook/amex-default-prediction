import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from features.build import create_categorical_train
from models.boosting import XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    lgb_trainer = XGBoostTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
