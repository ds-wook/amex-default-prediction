import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from features.build import (
    add_diff_features,
    add_trick_features,
    create_categorical_train,
)
from models.boosting import LightGBMTrainer
from utils import reduce_mem_usage, seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)
    # create dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    # train_x = train_x[cfg.features.selected_features]
    train_x = add_trick_features(train_x)
    train_x = add_diff_features(train_x)
    train_x = reduce_mem_usage(train_x)

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
