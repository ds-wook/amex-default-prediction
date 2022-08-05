import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from models.boosting import CatBoostTrainer


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    # create dataset
    train_x, train_y = load_train_dataset(cfg)

    # train model
    cb_trainer = CatBoostTrainer(config=cfg, metric=amex_metric)
    cb_trainer.train(train_x, train_y)

    # save model
    cb_trainer.save_model()


if __name__ == "__main__":
    _main()
