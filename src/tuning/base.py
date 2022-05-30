import warnings
from abc import ABCMeta, abstractclassmethod
from functools import partial
from pathlib import Path

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
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractclassmethod
    def _objective(self, trial: FrozenTrial, config: DictConfig) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
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
            "project": self.config.experiment.project,
            "entity": self.config.experiment.entity,
            "name": self.config.experiment.name,
            "reinit": self.config.experiment.reinit,
        }
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

        study = optuna.create_study(
            study_name=self.config.search.study_name,
            direction=self.config.search.direction,
            sampler=self._create_sampler(
                config=self.config.search.get("sampler", None),
            ),
            pruner=self._create_pruner(
                config=self.config.search.get("pruner", None),
            ),
        )

        # define callbacks
        objective = partial(self._objective, config=self.config)

        # optimize
        study.optimize(
            objective,
            n_trials=self.config.search.n_trials,
            callbacks=[wandbc],
        )

        wandb.run.summary["best accuracy"] = study.best_trial.value

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
        path = Path(get_original_cwd()) / self.config.search.path_name
        update_params = OmegaConf.load(path)

        update_params.model.params.update(study.best_trial.params)

        OmegaConf.save(update_params, path)

    def _create_sampler(self, config: DictConfig) -> BaseSampler:
        """
        Create sampler
        Args:
            sampler_mode: sampler mode
            seed: seed
        Returns:
            BaseSampler: sampler
        """
        # config update
        with open_dict(config):
            mode = config.pop("type")

        if mode == "random":
            sampler = RandomSampler(**config)
        elif mode == "tpe":
            sampler = TPESampler(**config)
        elif mode == "cma":
            sampler = CmaEsSampler(**config)
        else:
            raise ValueError(f"Unknown sampler mode: {mode}")

        return sampler

    def _create_pruner(self, config: DictConfig) -> BasePruner:
        """
        Create pruner
        Args:
            pruner_mode: pruner mode
            seed: seed
        Returns:
            HyperbandPruner: pruner
        """
        # config update
        with open_dict(config):
            mode = config.pop("type")

        if mode == "hyperband":
            pruner = HyperbandPruner(**config)
        elif mode == "median":
            pruner = MedianPruner(**config)
        elif mode == "nop":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner mode: {mode}")

        return pruner
