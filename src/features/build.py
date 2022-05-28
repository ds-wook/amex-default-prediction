from typing import List

import numpy as np
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import KFold
from tqdm import tqdm


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


def train_kfold_mean_encoding(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    cat_features: List[str],
) -> pd.DataFrame:
    for c in tqdm(cat_features):
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        target_mean = data_tmp.groupby(c)["target"].mean()

        # 학습 데이터 변환 후 값을 저장하는 배열 준비
        tmp = np.repeat(np.nan, train_x.shape[0])

        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        for train_idx, valid_idx in kf.split(train_x):
            # out of fold 로 각 범주형 목적변수 평균 계산
            target_mean = data_tmp.iloc[train_idx].groupby(c)["target"].mean()
            # 변환 후의 값을 날짜 배열에 저장
            tmp[valid_idx] = train_x[c].iloc[valid_idx].map(target_mean)

        train_x[c] = tmp

    return train_x, train_y


def test_kfold_mean_encoding(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    cat_features: List[str],
) -> pd.DataFrame:
    for c in tqdm(cat_features):
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        target_mean = data_tmp.groupby(c)["target"].mean()

        # 테스트 데이터의 카테고리 변경
        test_x[c] = test_x[c].map(target_mean)

    return test_x
