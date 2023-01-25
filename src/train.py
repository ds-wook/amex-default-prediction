import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from models.boosting import LightGBMTrainer
from utils.utils import seed_everything


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)
    # create dataset
    train_x, train_y = load_train_dataset(cfg)

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
