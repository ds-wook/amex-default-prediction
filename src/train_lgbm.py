import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/modeling/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    train_x = train_x.drop(columns=cfg.dataset.del_features)
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
