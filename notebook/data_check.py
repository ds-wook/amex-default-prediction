# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_lag_features_ensemble.csv")
preds2 = pd.read_csv("../output/lb_overfitting.csv")
# preds3 = pd.read_csv("../output/10fold_stacking_tabnet.csv")
# preds4 = pd.read_csv("../output/keras-cnn_sub.csv")
preds1.head()
# %%
preds1["prediction"] = (
    0.0005 * preds1["prediction"]
    + 0.9995 * preds2["prediction"]
)
preds1.head()

# %%
preds2.head()
# %%
preds1.to_csv("../output/final_submit_ensemble.csv", index=False)
# %%
df = preds1.copy()
df.head()
# %%
df["prediction2"] = preds2.prediction
df["prediction3"] = preds3.prediction
df["prediction4"] = preds4.prediction
# %%
df.corr()
# %%
scores_df = pd.read_csv("../input/scores_correlation.csv")
scores_df.head()
# %%
scores_df.sort_values("split_score", ascending=False)
# %%
scores_df[scores_df["gain_score"] > 0].shape
# %%
def get_score_correlation(
    actual_imp_df: pd.DataFrame, null_imp_df: pd.Series
) -> pd.DataFrame:
    correlation_scores = []
    for _f in actual_imp_df["feature"].unique():
        f_null_imps = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_gain"
        ].values
        f_act_imps = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_gain"
        ].values
        gain_score = (
            100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        )
        f_null_imps = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_split"
        ].values
        f_act_imps = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_split"
        ].values
        split_score = (
            100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        )
        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(
        correlation_scores, columns=["feature", "split_score", "gain_score"]
    )

    return corr_scores_df


# %%
shap_df = pd.read_csv("../input/shap_importance.csv")
shap_df.head()
# %%
shap_df.values.tolist()
# %%
train = pd.read_parquet(
    "../input/amex-bruteforce-features/train_bruteforce_features.parquet"
)
train.head()

# %%
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
            # df[f"{col}_first_mean_diff"] = df[f"{col}_first"] - df[f"{col}_mean"]
        except Exception:
            pass

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    num_cols = [col for col in df.columns if "last" in col]
    num_cols = [col[:-5] for col in num_cols if "round" not in col]

    # Lag Features
    for col in num_cols:
        try:
            df[f"{col}_last_first_diff"] = df[f"{col}_last"] - df[f"{col}_first"]
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

# %%
train_x = train.drop(columns=["customer_ID", "S_2"])
train_x = add_trick_features(train_x)
train_x = add_diff_features(train_x)
train_x = add_lag_features(train_x)
train_x.head()
# %%
