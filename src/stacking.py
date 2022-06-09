from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from evaluation.evaluate import amex_metric
from models.boosting import LightGBMTrainer
from models.infer import load_model


@hydra.main(config_path="../config/", config_name="stacking.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.dataset.path
    train = pd.read_csv(path / cfg.dataset.train)

    credit_select = load_model(cfg, cfg.output.credit_select)
    credit = load_model(cfg, cfg.output.credit)
    trick_credit = load_model(cfg, cfg.output.trick_credit)
    trick_select = load_model(cfg, cfg.output.trick_select)

    train["credit_select"] = credit_select.oof_preds
    train["credit"] = credit.oof_preds
    train["trick_credit"] = trick_credit.oof_preds
    train["trick_select"] = trick_select.oof_preds

    train_x = train.drop(columns=[cfg.dataset.drop_features, cfg.dataset.target])
    train_y = train[cfg.dataset.target]
    # train
    xgb_trainer = LightGBMTrainer(config=cfg, metric=amex_metric)
    xgb_trainer.train(train_x, train_y)

    # save model
    xgb_trainer.save_model()


if __name__ == "__main__":
    _main()
