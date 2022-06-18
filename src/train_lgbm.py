import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from features.build import create_categorical_train
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    # create dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    train_x = train_x[cfg.features.selected_features]

    train_x = train_x.fillna(-127)

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
