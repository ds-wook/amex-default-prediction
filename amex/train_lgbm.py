import hydra
from omegaconf import DictConfig

from amex.data.dataset import load_train_dataset
from amex.evaluation.evaluate import amex_metric
from amex.features.build import create_categorical_train
from amex.models.boosting import LightGBMTrainer
from amex.utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.models.params.seed)
    # create dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = create_categorical_train(train_x, cfg)

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
