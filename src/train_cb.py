import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from features.build import create_categorical_train
from models.boosting import CatBoostTrainer


@hydra.main(config_path="../config/modeling/", config_name="cb.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    cb_trainer = CatBoostTrainer(config=cfg, metric=amex_metric)
    cb_trainer.train(train_x, train_y)
    cb_trainer.save_model()


if __name__ == "__main__":
    _main()
