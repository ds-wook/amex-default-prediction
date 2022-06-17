import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from features.build import create_categorical_train
from models.boosting import CatBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    # create dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)
    train_x = train_x[cfg.dataset.selected_features]
    train_x.fillna(-127, inplace=True)

    # train model
    cb_trainer = CatBoostTrainer(config=cfg, metric=amex_metric)
    cb_trainer.train(train_x, train_y)

    # save model
    cb_trainer.save_model()


if __name__ == "__main__":
    _main()
