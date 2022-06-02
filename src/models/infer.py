import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset
from features.build import make_trick
from models.base import ModelResult


def load_model(model_name: str) -> ModelResult:
    """
    Load model
    Args:
        model_name: model name
    Returns:
        ModelResult object
    """
    model_path = Path(get_original_cwd()) / model_name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result


def predict(result: ModelResult, config: DictConfig) -> List[np.ndarray]:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """
    preds_proba = []

    for num in tqdm(range(10)):
        test_sample = load_test_dataset(config, num)
        logging.info(f"Predicting...{test_sample.shape}")
        test_sample = make_trick(test_sample)

        for model in result.models.values():
            preds = model.predict_proba(test_sample)[:, 1]
            preds_proba += list(preds)

    return preds_proba
