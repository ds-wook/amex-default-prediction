import eli5
import pandas as pd
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def test_selected_permutation_importances(
    data: pd.DataFrame, label: pd.Series, config: DictConfig
) -> pd.DataFrame:
    """
    Permutation importance
    Args:
        data: dataframe
        label: label
        config: config
    Returns:
        imp_df: importance dataframe
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        label,
        test_size=0.3,
        random_state=config.model.params.seed,
        stratify=label,
    )
    print(X_train.isna().sum().sum())
    print(X_test.isna().sum().sum())
    print(y_train.isna().sum().sum())
    print(y_test.isna().sum().sum())
    model = LGBMClassifier(random_state=config.model.params.seed)
    model.fit(X_train, y_train)
    perm_lgbm = PermutationImportance(model, random_state=config.model.params.seed).fit(
        X_test, y_test
    )

    pi_features = eli5.explain_weights_df(
        perm_lgbm, feature_names=X_train.columns.tolist()
    )

    return pi_features
