import gc
from typing import List

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def categorical_train_encoding(
    train: pd.DataFrame, cat_features: List[str]
) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    le_encoder = LabelEncoder()
    for categorical_feature in tqdm(cat_features):
        train[categorical_feature] = le_encoder.fit_transform(
            train[categorical_feature]
        )
    return train


def categorical_test_encoding(
    train: pd.DataFrame, test: pd.DataFrame, cat_features: List[str]
) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    le_encoder = LabelEncoder()
    for categorical_feature in tqdm(cat_features):
        train[categorical_feature] = le_encoder.fit_transform(train[categorical_feature])
        test[categorical_feature] = le_encoder.transform(test[categorical_feature])

    del train

    return test


def count_train_encoding(train: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    encoder = CountEncoder(cols=cat_features)
    train = encoder.fit_transform(train)
    return train


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


def create_avg_features(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Create average features
    Args:
        df: dataframe
        config: config file
    Returns:
        dataframe
    """
    cid = pd.Categorical(df.pop("customer_ID"), ordered=True)
    last = cid != np.roll(cid, -1)  # mask for last statement of every customer

    if config.dataset.is_train:
        target = df.loc[last, config.dataset.target]
        gc.collect()

    df_avg = (
        df[config.dataset.features_avg]
        .groupby(cid)
        .mean()
        .rename(columns={f: f"{f}_avg" for f in config.dataset.features_avg})
    )
    gc.collect()

    df = (
        df.loc[last, config.dataset.features_last]
        .rename(columns={f: f"{f}_last" for f in config.dataset.features_last})
        .set_index(np.asarray(cid[last]))
    )
    gc.collect()

    df = pd.concat([df, df_avg], axis=1)
    del df_avg, cid, last

    if config.dataset.is_train:
        return df, target

    return df
