from typing import List

import pandas as pd
from category_encoders.ordinal import OrdinalEncoder


def categorical_train_encoding(train: pd.DataFrame, cat_col: List[str]) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    encoder = OrdinalEncoder(cols=cat_col)
    train = encoder.fit_transform(train)
    return train


def categorical_test_encoding(
    train: pd.DataFrame, test: pd.DataFrame, cat_col: List[str]
) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    encoder = OrdinalEncoder(cols=cat_col)
    train = encoder.fit_transform(train)
    test = encoder.transform(test)
    return train, test
