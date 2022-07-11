from typing import Union

import numpy as np
import pandas as pd


def last_2(series: pd.Series) -> Union[int, float]:
    return series.values[-2] if len(series.values) >= 2 else np.nan


def last_3(series: pd.Series) -> Union[int, float]:
    return series.values[-3] if len(series.values) >= 3 else np.nan


def median(x):
    return np.median(x)


def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def variance(x):
    return np.var(x)


def skewness(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def standard_deviation(x):
    return np.std(x)


def large_standard_deviation(x):
    if (np.max(x) - np.min(x)) == 0:
        return np.nan
    else:
        return np.std(x) / (np.max(x) - np.min(x))


def variance_std_ratio(x):
    y = np.var(x)
    if y != 0:
        return y / np.sqrt(y)
    else:
        return np.nan


def ratio_beyond_r_sigma(x, r):
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size


def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference


def has_duplicate_max(x):
    return np.sum(x == np.max(x)) >= 2


def has_duplicate_min(x):
    return np.sum(x == np.min(x)) >= 2


def has_duplicate(x):
    return x.size != np.unique(x).size


def count_duplicate_max(x):
    return np.sum(x == np.max(x))


def count_duplicate_min(x):
    return np.sum(x == np.min(x))


def count_duplicate(x):
    return x.size - np.unique(x).size


def sum_values(x):
    if len(x) == 0:
        return 0
    return np.sum(x)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def realized_abs_skew(series):
    return np.power(np.abs(np.sum(series**3)), 1 / 3)


def realized_skew(series):
    return np.sign(np.sum(series**3)) * np.power(np.abs(np.sum(series**3)), 1 / 3)


def realized_vol_skew(series):
    return np.power(np.abs(np.sum(series**6)), 1 / 6)


def realized_quarticity(series):
    return np.power(np.sum(series**4), 1 / 4)


def count_unique(series):
    return len(np.unique(series))


def count(series):
    return series.size


# drawdons functions are mine
def maximum_drawdown(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def maximum_drawup(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    series = -series
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def drawdown_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j = k
    else:
        j = np.argmax(series[:i])
    return k - j


def drawup_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    series = -series
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j = k
    else:
        j = np.argmax(series[:i])
    return k - j


def max_over_min(series):
    if len(series) < 2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.max(series) / np.min(series)


def mean_n_absolute_max(x, number_of_maxima=1):
    """Calculates the arithmetic mean of the n absolute maximum values of the time series."""
    assert (
        number_of_maxima > 0
    ), f" number_of_maxima={number_of_maxima} which is not greater than 1"

    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.NaN


def count_above(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x >= t) / len(x)


def count_below(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x <= t) / len(x)


def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))


def mean_change(x):
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN


def mean_second_derivative_central(x):
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN


def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.NaN


def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))


def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size


def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size


def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


# Test non-consecutive non-reoccuring values ?
def percentage_of_reoccurring_values_to_all_values(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])


def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size


def sum_of_reoccurring_values(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def quantile(x, q):
    if len(x) == 0:
        return np.NaN
    return np.quantile(x, q)


# crossing the mean ? other levels ?
def number_crossing_m(x, m):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > m
    return np.where(np.diff(positive))[0].size


def absolute_maximum(x):
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN


def value_count(x, value):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if np.isnan(value):
        return np.isnan(x).sum()
    else:
        return x[x == value].size


def range_count(x, min, max):
    return np.sum((x >= min) & (x < max))


def mean_diff(x):
    return np.nanmean(np.diff(x.values))
