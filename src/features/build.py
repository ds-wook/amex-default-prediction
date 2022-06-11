import gc
import pickle
from pathlib import Path

import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def create_cb_encoder_train(
    train_x: pd.DataFrame, train_y: pd.Series, config: DictConfig
) -> pd.DataFrame:
    """
    Target encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder
    cb_encoder = CatBoostEncoder(config.dataset.cat_features)
    train_x = cb_encoder.fit_transform(train_x, train_y)

    with open(path / "catboost_encoder.pkl", "wb") as f:
        pickle.dump(cb_encoder, f)

    return train_x


def create_target_encoder_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Target encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder
    cb_encoder = pickle.load(open(path / "catboost_encoder.pkl", "rb"))
    test = cb_encoder.transform(test)
    return test


def create_target_encoder_train(
    train_x: pd.DataFrame, train_y: pd.Series, config: DictConfig
) -> pd.DataFrame:
    """
    Target encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder
    te_encoder = TargetEncoder(config.dataset.cat_features)
    train_x = te_encoder.fit_transform(train_x, train_y)

    with open(path / "target_encoder.pkl", "wb") as f:
        pickle.dump(te_encoder, f)

    return train_x


def create_cb_encoder_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Target encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder
    te_encoder = pickle.load(open(path / "target_encoder.pkl", "rb"))
    test = te_encoder.transform(test)
    return test


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created

    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in all_cols if col not in cat_features]

    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["last", "nunique", "count"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]

    df = pd.concat([df_num_agg, df_cat_agg], axis=1)
    del df_num_agg, df_cat_agg
    gc.collect()

    print("shape after engineering", df.shape)

    return df


def make_trick(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create nan feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """

    for col in df.columns:
        if "float" in df[col].dtype.name:
            df[col] = df[col].round(decimals=2)

    return df


def add_after_pay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are only available after payment.
    Args:
        df: DataFrame with customer_ID as index.
    Returns:
        DataFrame with customer_ID as index and additional features.
    """
    df = df.copy()
    before_cols = ["B_11", "B_14", "B_17", "D_39", "D_131", "S_16", "S_23"]
    after_cols = ["P_2", "P_3"]

    # after pay features
    after_pay_features = []
    for b_col in before_cols:
        for a_col in after_cols:
            if b_col in df.columns:
                df[f"{b_col}-{a_col}"] = df[b_col] - df[a_col]
                df[f"{b_col}/{a_col}"] = df[b_col] / df[a_col]
                after_pay_features.append(f"{b_col}-{a_col}")
                after_pay_features.append(f"{b_col}/{a_col}")

    df_after_agg = df.groupby("customer_ID")[after_pay_features].agg(
        ["mean", "std", "last"]
    )
    df_after_agg.columns = ["_".join(x) for x in df_after_agg.columns]

    return df_after_agg
