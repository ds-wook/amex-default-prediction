import logging
from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from shap import TreeExplainer


def select_features(
    train: pd.DataFrame, label: pd.Series, test: pd.DataFrame
) -> List[str]:
    model = LGBMClassifier(random_state=42, learning_rate=0.1)
    logging.info(f"{model.__class__.__name__} Train Start!")
    model.fit(train, label)
    explainer = TreeExplainer(model)

    shap_values = explainer.shap_values(test)
    shap_sum = np.abs(shap_values).mean(axis=1).sum(axis=0)

    importance_df = pd.DataFrame([test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    importance_df = importance_df.query("shap_importance != 0")
    boosting_shap_col = importance_df.column_name.values.tolist()
    logging.info(f"Select {len(boosting_shap_col)} of {len(train.columns)}")

    return boosting_shap_col
