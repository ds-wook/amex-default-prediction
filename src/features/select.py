from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from sklearn.model_selection import train_test_split


def select_features(train: pd.DataFrame, target: pd.Series) -> List[str]:

    x_train, x_test, y_train, y_test = train_test_split(
        train, target, test_size=0.2, random_state=42, stratify=target
    )
    model = LGBMClassifier(random_state=42)
    print(f"{model.__class__.__name__} Train Start!")
    model.fit(x_train, y_train)
    explainer = TreeExplainer(model)

    shap_values = explainer.shap_values(x_test)
    shap_sum = np.abs(shap_values).mean(axis=1).sum(axis=0)

    importance_df = pd.DataFrame([x_test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    importance_df = importance_df.query("shap_importance == 0")
    boosting_shap_col = importance_df.column_name.values.tolist()

    return boosting_shap_col
