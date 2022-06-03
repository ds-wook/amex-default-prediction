import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from models.base import ModelResult


def load_model(config: DictConfig) -> ModelResult:
    """
    Load model
    Args:
        model_name: model name
    Returns:
        ModelResult object
    """
    model_path = Path(get_original_cwd()) / config.model.path / config.model.name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result


def predict(result: ModelResult, test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(result.models)
    preds_proba = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds):
        preds_proba += model.predict_proba(test_x)[:, 1] / folds

    assert len(preds_proba) == len(test_x)

    return preds_proba
