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


def last_2(series: pd.Series) -> Union[int, float]:
    return series.values[-2] if len(series.values) >= 2 else np.nan


def last_3(series: pd.Series) -> Union[int, float]:
    return series.values[-3] if len(series.values) >= 3 else np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    time_features = (
        "D_39,D_41,D_47,D_45,D_46,D_48,D_54,D_59,D_61,D_62,D_75,D_96,D_105,D_112,D_124,"
        "S_3,S_7,S_19,S_23,S_26,P_2,P_3,B_2,B_3,B_4,B_5,B_7,B_9,B_20,R_1,R_3,R_13,R_18"
    )
    time_features = time_features.split(",")

    for col in tqdm(time_features):
        df[f"{col}_diff"] = df.groupby("customer_ID")[col].diff()

    return df


def get_difference(data: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis=0)
    # Transform to dataframe
    df1 = pd.DataFrame(
        df1, columns=[col + "_diff1" for col in df[num_features].columns]
    )
    # Add customer id
    df1["customer_ID"] = customer_ids
    return df1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    features = df.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    cat_features = (
        "B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_66, D_68"
    )

    cat_features = cat_features.split(", ")

    num_features = [col for col in features if col not in cat_features]

    print("Starting training feature engineer...")
    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["count", "last", "nunique"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    # Transform float64 columns to float32
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == "float64"].index)
    for col in tqdm(cols):
        df_num_agg[col] = df_num_agg[col].astype(np.float32)

    # Transform int64 columns to int32
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        df_cat_agg[col] = df_cat_agg[col].astype(np.int32)

    # Get the difference
    df_diff = get_difference(df, num_features)
    df = df_num_agg.merge(df_cat_agg, how="inner", on="customer_ID").merge(
        df_diff, how="inner", on="customer_ID"
    )

    del df_num_agg, df_cat_agg, df_diff
    gc.collect()

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
    num_cols = [col for col in num_cols if "last" in col]

    for col in num_cols:
        df[col + "_round2"] = df[col].round(2)

    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are only available after payment.
    Args:
        df: DataFrame with customer_ID as index.
    Returns:
        DataFrame with customer_ID as index and additional features.
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


def feature_filter(data: pd.DataFrame, threshold: float = 0.1) -> List[str]:
    features = data.columns
    filtered_features = []
    for feature in features:
        if data[feature].isnull().sum() < threshold:
            filtered_features.append(feature)
    return filtered_features


def feature_correlation(
    data: pd.DataFrame, target: pd.Series, threshold: float = 0.1
) -> List[str]:
    data = pd.concat([data, target], axis=1)
    correlations = data.corr()["target"].drop("target")

    # Filter the features with correlation to the target less than threshold
    filtered_features = correlations[abs(correlations) < threshold].index.tolist()

    # save memory
    del data
    gc.collect()

    return filtered_features


def fill_missing_values(
    data: pd.DataFrame, imputation_method: str = "median"
) -> pd.DataFrame:
    data_copy = data.copy()
    for column in data_copy.columns:
        if data_copy[column].dtype == np.dtype("O"):
            data_copy[column] = data_copy[column].fillna(
                data_copy[column].mode().iloc[0]
            )
        else:
            if imputation_method == "median":
                data_copy[column] = data_copy[column].fillna(data_copy[column].median())
            elif imputation_method == "mean":
                data_copy[column] = data_copy[column].fillna(data_copy[column].mean())
    return data_copy
