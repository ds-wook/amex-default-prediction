import gc
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()


def create_categorical_train(train: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    le_encoder = LabelEncoder()

    for cat_feature in tqdm(config.dataset.cat_features):
        train[cat_feature] = le_encoder.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(le_encoder, f)

    return train


def create_categorical_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    for cat_feature in tqdm(config.dataset.cat_features):
        le_encoder = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        test[cat_feature] = le_encoder.transform(test[cat_feature])
        gc.collect()

    return test


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # Get the difference between last and mean
    num_cols = [col for col in df.columns if "last" in col]
    num_cols = [col[:-5] for col in num_cols if "round" not in col]

    for col in num_cols:
        try:
            df[f"{col}_last_mean_diff"] = df[f"{col}_last"] - df[f"{col}_mean"]
        except Exception:
            pass

    return df


def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # Get the difference between last and mean
    num_cols = [col for col in df.columns if "last" in col]
    num_cols = [col[:-5] for col in num_cols if "round" not in col]

    for col in num_cols:
        try:
            df[f"{col}_last_mean_rate"] = df[f"{col}_last"] / df[f"{col}_mean"]
        except Exception:
            pass

    return df


def get_difference(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
        num_features: list of numerical features
    Returns:
        dataframe
    """

    df_diff = (
        df.loc[:, num_features + ["customer_ID"]]
        .groupby(["customer_ID"])
        .progress_apply(
            lambda x: np.diff(x.values[-2:, :], axis=0).squeeze().astype(np.float32)
        )
    )
    cols = [col + "_diff1" for col in df[num_features].columns]
    df_diff = pd.DataFrame(df_diff.values.tolist(), columns=cols, index=df_diff.index)
    df_diff.reset_index(inplace=True)

    return df_diff


def get_gradient(x: np.ndarray) -> Union[pd.DataFrame, List]:
    try:
        return np.gradient(x, axis=0)[-1]
    except ValueError:
        return []


def add_pay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pay features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # compute "after pay" features
    before_cols = ["B_9", "B_11", "B_14", "B_17", "D_39", "D_131", "S_16", "S_23"]
    after_cols = ["P_2", "P_3"]
    for bcol in before_cols:
        for pcol in after_cols:
            if bcol in df.columns:
                df[f"{bcol}-{pcol}"] = df[bcol] - df[pcol]

    return df


def add_fft_features(df: pd.DataFrame, time_features: List[str]) -> pd.DataFrame:
    """
    Create fft features
    Args:
        df: datafram
        time_features: list of time features
    Returns:
        dataframe
    """
    df_fft = (
        df.loc[:, time_features + ["customer_ID"]]
        .groupby(["customer_ID"])
        .progress_apply(lambda x: np.fft.fft(x, axis=0)[-1])
    )
    cols = [col + "_fft" for col in df[time_features].columns]
    df_fft = pd.DataFrame(df_fft.values.tolist(), columns=cols, index=df_fft.index)
    df_fft.reset_index(inplace=True)
    return df_fft


def add_gradient_features(df: pd.DataFrame, time_features: List[str]) -> pd.DataFrame:
    """
    Create gradient features
    Args:
        df: dataframe
        time_features: list of time features
    Returns:
        dataframe
    """
    df_grad = (
        df.loc[:, time_features + ["customer_ID"]]
        .groupby(["customer_ID"])
        .progress_apply(lambda x: get_gradient(x))
    )
    cols = [col + "_gradient" for col in df[time_features].columns]
    df_grad = pd.DataFrame(df_grad.values.tolist(), columns=cols, index=df_grad.index)
    df_grad.reset_index(inplace=True)
    return df_grad


def add_trick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create nan feature
    Args:
        df: dataframe
    Returns:stacking neural network
        dataframe
    """
    num_cols = df.dtypes[
        (df.dtypes == "float32") | (df.dtypes == "float64")
    ].index.to_list()
    num_cols = [col for col in num_cols if "last" in col]

    for col in num_cols:
        df[col + "_round2"] = df[col].round(2)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = (
        "B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_66, D_68"
    )

    cat_features = cat_features.split(", ")

    time_features = (
        "D_39,D_41,D_47,D_45,D_46,D_48,D_54,D_59,D_61,D_62,D_75,D_96,D_105,D_112,D_124,"
        "S_3,S_7,S_19,S_23,S_26,P_2,P_3,B_2,B_3,B_4,B_5,B_7,B_9,B_20,R_1,R_3,R_13,R_18"
    )
    time_features = time_features.split(",")
    num_features = [col for col in all_cols if col not in cat_features]

    # Get the difference
    df_diff = get_difference(df, num_features)

    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["last", "nunique", "count"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    # gradient features
    df_grad_agg = add_gradient_features(df, time_features)

    # Transform int64 columns to int32
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == "float64"].index)
    df_num_agg.loc[:, cols] = df_num_agg.loc[:, cols].progress_apply(
        lambda x: x.astype(np.float32)
    )

    # Transform int64 columns to int32
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == "int64"].index)
    df_cat_agg.loc[:, cols] = df_cat_agg.loc[:, cols].progress_apply(
        lambda x: x.astype(np.int32)
    )

    df = (
        df_num_agg.merge(df_cat_agg, how="inner", on="customer_ID")
        .merge(df_grad_agg, how="inner", on="customer_ID")
        .merge(df_diff, how="inner", on="customer_ID")
    )

    del df_num_agg, df_cat_agg, df_diff, df_grad_agg
    gc.collect()

    return df
