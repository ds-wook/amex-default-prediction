import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset, load_train_dataset_parquet
from evaluation.evaluate import amex_metric
from models.boosting import LightGBMTrainer
from utils import reduce_mem_usage, seed_everything


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    seed_everything(cfg.model.params.seed)

    # create dataset
    train_x, train_y = (
        load_train_dataset(cfg)
        if cfg.dataset.type == "pkl"
        else load_train_dataset_parquet(cfg)
    )
    # train_x = train_x[cfg.features.selected_features]
    train_x = reduce_mem_usage(train_x)

    # train model
    lgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
