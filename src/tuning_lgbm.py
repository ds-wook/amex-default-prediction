import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from evaluation.evaluate import amex_metric
from tuning.boosting import LightGBMTuner
from utils.utils import reduce_mem_usage, seed_everything


@hydra.main(config_path="../config/", config_name="search")
def _main(cfg: DictConfig):
    seed_everything(cfg.dataset.seed)

    # create dataset
    train_x, train_y = load_train_dataset(cfg)
    train_x = reduce_mem_usage(train_x)

    # tuning model
    lgb_tuner = LightGBMTuner(
        train_x=train_x, train_y=train_y, config=cfg, metric=amex_metric
    )
    study = lgb_tuner.build_study()

    # save hyparparameters
    lgb_tuner.save_hyperparameters(study)


if __name__ == "__main__":
    _main()
