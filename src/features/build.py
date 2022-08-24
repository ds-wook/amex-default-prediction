import gc
import pickle
from pathlib import Path

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
        config: config
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    le_encoder = LabelEncoder()

    for cat_feature in tqdm(config.features.cat_features):
        train[cat_feature] = le_encoder.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(le_encoder, f)

    return train


def create_categorical_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        config: config
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    for cat_feature in tqdm(config.features.cat_features):
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


def add_trick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create nan feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    num_cols = df.dtypes[
        (df.dtypes == "float32") | (df.dtypes == "float64")
    ].index.to_list()
    num_cols = [col for col in num_cols if "last" in col or "first" in col]

    for col in num_cols:
        df[col + "_round2"] = df[col].round(2)

    return df


def add_customized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customized features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    df["c_PD_239"] = df["D_39_last"] / ((df["P_2_last"] * -1) + 0.0001)
    df["c_PB_29"] = (df["P_2_last"] * -1) / (df["B_9_last"] + 0.0001)
    df["c_PR_21"] = (df["P_2_last"] * -1) / (df["R_1_last"] + 0.0001)

    df["c_BBBB"] = (df["B_9_last"] + 0.001) / (
        df["B_23_last"] + df["B_3_last"] + 0.0001
    )
    df["c_BBBB1"] = (df["B_33_last"] * -1) + (
        df["B_18_last"] * (-1) + df["S_25_last"] * (1) + 0.0001
    )
    df["c_BBBB2"] = df["B_19_last"] + df["B_20_last"] + df["B_4_last"] + 0.0001

    df["c_RRR0"] = (df["R_3_last"] + 0.001) / (df["R_2_last"] + df["R_4_last"] + 0.0001)
    df["c_RRR1"] = (df["D_62_last"] + 0.001) / (
        df["D_112_last"] + df["R_27_last"] + 0.0001
    )

    df["c_PD_348"] = df["D_48_last"] / (df["P_3_last"] + 0.0001)
    df["c_PD_355"] = df["D_55_last"] / (df["P_3_last"] + 0.0001)

    df["c_PD_439"] = df["D_39_last"] / (df["P_4_last"] + 0.0001)
    df["c_PB_49"] = df["B_9_last"] / (df["P_4_last"] + 0.0001)
    df["c_PR_41"] = df["R_1_last"] / (df["P_4_last"] + 0.0001)

    return df
