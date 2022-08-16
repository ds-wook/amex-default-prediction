import warnings
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization import plot_optimization_history, plot_param_importances

warnings.filterwarnings("ignore")


class BaseTuner(metaclass=ABCMeta):
    def __init__(
        self, config: DictConfig, metric: Callable[[np.ndarray, np.ndarray], float]
    ):
        self.config = config
        self.metric = metric

    @abstractclassmethod
    def _objective(self, trial: FrozenTrial) -> float:
        """
        Objective function
        Args:
            trial: trial object
        Returns:
            metric score
        """
        raise NotImplementedError

    def build_study(self, verbose: bool = False) -> Study:
        """
        Build study
        Args:
            study_name: study name
        Returns:
            study
        """
        wandb_kwargs = {
            "project": self.config.logger.project,
            "entity": self.config.logger.entity,
            "name": self.config.logger.name,
        }
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

        study = optuna.create_study(
            study_name=self.config.tuning.search.study_name,
            direction=self.config.tuning.search.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
        )

        # optimize
        study.optimize(
            self._objective,
            n_trials=self.config.tuning.search.n_trials,
            callbacks=[wandbc],
        )

        wandb.run.summary["best_accuracy"] = study.best_trial.value

        wandb.log(
            {
                "optuna_optimization_history": plot_optimization_history(study),
                "optuna_param_importances": plot_param_importances(study),
            }
        )

        wandb.finish()

        return study

    def save_hyperparameters(self, study: Study) -> None:
        """
        Save best hyperparameters to yaml file
        Args:
            study: study best hyperparameter object.
        """
        path = Path(get_original_cwd()) / self.config.tuning.search.path_name
        update_params = OmegaConf.load(path)

        update_params.model.params.update(study.best_trial.params)

        OmegaConf.save(update_params, path)

    def _create_sampler(self) -> BaseSampler:
        """
        Create sampler
        Args:
            sampler_mode: sampler mode
            seed: seed
        Returns:
            BaseSampler: sampler
        """
        # config update
        with open_dict(self.config.tuning.search):
            mode = self.config.tuning.search.sampler.pop("type")

        if mode == "random":
            sampler = RandomSampler(**self.config.tuning.search.sampler)
        elif mode == "tpe":
            sampler = TPESampler(**self.config.tuning.search.sampler)
        elif mode == "cma":
            sampler = CmaEsSampler(**self.config.tuning.search.sampler)
        else:
            raise ValueError(f"Unknown sampler mode: {mode}")

        return sampler

    def _create_pruner(self) -> BasePruner:
        """
        Create pruner
        Args:
            pruner_mode: pruner mode
            seed: seed
        Returns:
            HyperbandPruner: pruner
        """
        # config update
        with open_dict(self.config.tuning.search):
            mode = self.config.tuning.search.pruner.pop("type")

        if mode == "hyperband":
            pruner = HyperbandPruner(**self.config.tuning.search.pruner)
        elif mode == "median":
            pruner = MedianPruner(**self.config.tuning.search.pruner)
        elif mode == "nop":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner mode: {mode}")

        return pruner
