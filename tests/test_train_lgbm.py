import hydra
from omegaconf import DictConfig

from test_data.test_dataset import load_train_dataset
from test_evaluation.test_evaluate import amex_metric
from test_models.test_boosting import LightGBMTrainer
from utils.utils import seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)
    # create dataset
    train_x, train_y = load_train_dataset(cfg)

    # # select features
    # train_x = train_x[cfg.features.selected_features]

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)


if __name__ == "__main__":
    _main()
