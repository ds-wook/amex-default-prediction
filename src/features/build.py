from typing import List

import pandas as pd
from category_encoders.ordinal import OrdinalEncoder


def categorical_encoding(
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
