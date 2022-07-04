import gc
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
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


def last_2(series: pd.Series) -> Union[int, float]:
    return series.values[-2] if len(series.values) >= 2 else np.nan


def last_3(series: pd.Series) -> Union[int, float]:
    return series.values[-3] if len(series.values) >= 3 else np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    time_features = (
        "S_17, D_104, B_30, D_68, R_14, S_13, D_138, D_74, D_142, D_66, D_45, D_143, "
        "D_103, D_51, R_5, D_132, P_2, D_47, S_7, D_61, B_20, R_21, B_38, D_137, R_16, "
        "D_71, D_78, R_10, S_22, D_128, D_122, B_17, D_84, D_134, D_114, D_88, D_107, "
        "D_63, D_52, S_6, B_40, D_127, B_24, R_23, D_49, D_56, D_65, D_118, D_81, "
        "D_75, D_106, D_129, D_59, S_18, B_21, D_102, D_53, B_9, D_133, D_96, B_25, "
        "B_7, S_27, D_141, R_8, D_109, S_12, D_120, D_46, S_25, B_27, D_93, B_16, "
        "D_58, S_15, S_20, B_14, S_24, D_54, D_69, D_62, D_87, D_131, R_26, R_25, "
        "D_91, B_31, R_4, D_108, B_5, B_28, B_19, D_144, D_41, B_10, D_43, D_44, B_32, "
        "D_130, S_11, B_2, D_119, D_89, R_6, S_5, B_26, R_11, B_36, D_64, D_42, B_37, "
        "R_3, B_8, R_2, D_123, D_145, S_8, B_41, R_22, B_42, D_ 117, D_48, D_124, "
        "D_55, D_94, B_4, R_1, R_28, D_110, D_72, D_92, S_16, D_139, D_135, D_136, "
        "B_12, R_13, D_116, D_125, D_39, B_15, D_86, B_33, D_80, B_6, R_20, D_105, "
        "P_3, S_3, D_112, B_1, R_17, D_113, D_121, D_50, S_26, B_18, R_12, S_19, "
        "D_126, B_23, D_76, S_9, S_23, D_82, D_60, R_7, R_27, D_111, R_18, D_70, "
        "R_19, D_77, D_83, B_13, D_115, B_22, R_24, P_4, R_9, R_15, D_79, B_3, B_39, "
        "B_11, D_73, D_140"
    )
    time_features = time_features.split(", ")

    for col in time_features:
        df[f"{col}_diff"] = df.groupby("customer_ID")[col].diff()
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
        "S_17, D_104, B_30, D_68, R_14, S_13, D_138, D_74, D_142, D_66, D_45, D_143, "
        "D_103, D_51, R_5, D_132, P_2, D_47, S_7, D_61, B_20, R_21, B_38, D_137, R_16, "
        "D_71, D_78, R_10, S_22, D_128, D_122, B_17, D_84, D_134, D_114, D_88, D_107, "
        "D_63, D_52, S_6, B_40, D_127, B_24, R_23, D_49, D_56, D_65, D_118, D_81, "
        "D_75, D_106, D_129, D_59, S_18, B_21, D_102, D_53, B_9, D_133, D_96, B_25, "
        "B_7, S_27, D_141, R_8, D_109, S_12, D_120, D_46, S_25, B_27, D_93, B_16, "
        "D_58, S_15, S_20, B_14, S_24, D_54, D_69, D_62, D_87, D_131, R_26, R_25, "
        "D_91, B_31, R_4, D_108, B_5, B_28, B_19, D_144, D_41, B_10, D_43, D_44, B_32, "
        "D_130, S_11, B_2, D_119, D_89, R_6, S_5, B_26, R_11, B_36, D_64, D_42, B_37, "
        "R_3, B_8, R_2, D_123, D_145, S_8, B_41, R_22, B_42, D_ 117, D_48, D_124, "
        "D_55, D_94, B_4, R_1, R_28, D_110, D_72, D_92, S_16, D_139, D_135, D_136, "
        "B_12, R_13, D_116, D_125, D_39, B_15, D_86, B_33, D_80, B_6, R_20, D_105, "
        "P_3, S_3, D_112, B_1, R_17, D_113, D_121, D_50, S_26, B_18, R_12, S_19, "
        "D_126, B_23, D_76, S_9, S_23, D_82, D_60, R_7, R_27, D_111, R_18, D_70, "
        "R_19, D_77, D_83, B_13, D_115, B_22, R_24, P_4, R_9, R_15, D_79, B_3, B_39, "
        "B_11, D_73, D_140"
    )
    time_features = time_features.split(", ")
    time_features = [f"{col}_diff" for col in time_features]

    num_features = [col for col in all_cols if col not in cat_features + time_features]

    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["last", "nunique", "count"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]

    df_time_agg = df.groupby("customer_ID")[time_features].agg(["last", last_2, last_3])
    df_time_agg.columns = ["_".join(x) for x in df_time_agg.columns]

    df = pd.concat([df_num_agg, df_cat_agg, df_time_agg], axis=1)

    del df_num_agg, df_cat_agg, df_time_agg
    gc.collect()

    return df


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
                after_pay_features.append(f"{b_col}-{a_col}")

    df_after_agg = df.groupby("customer_ID")[after_pay_features].agg(
        ["mean", "std", "last"]
    )
    df_after_agg.columns = ["_".join(x) for x in df_after_agg.columns]

    return df_after_agg


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
