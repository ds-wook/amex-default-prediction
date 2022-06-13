import argparse

import pandas as pd

from data.dataset import split_dataset
from features.build import add_after_pay_features, build_features


def main(args: argparse.ArgumentParser):
    train = pd.read_parquet(args.path + "train.parquet")
    split_ids = split_dataset(train.customer_ID.unique(), 5)

    for (i, ids) in enumerate(split_ids):
        train_sample = train[train.customer_ID.isin(ids)]
        train_pay_agg = add_after_pay_features(train_sample)
        train_agg = build_features(train_sample)
        train_agg = pd.concat([train_agg, train_pay_agg], axis=1)
        print(i, train_agg.shape)
        train_agg.to_pickle(
            f"input/amex-pay-features/train_pay_features_part_{i}.pkl",
            compression="gzip",
        )
        del train_agg, train_pay_agg

    train_sample0 = pd.read_pickle(
        "input/amex-pay-features/train_pay_features_part_0.pkl", compression="gzip"
    )
    train_sample1 = pd.read_pickle(
        "input/amex-pay-features/train_pay_features_part_1.pkl", compression="gzip"
    )
    train_sample2 = pd.read_pickle(
        "input/amex-pay-features/train_pay_features_part_2.pkl", compression="gzip"
    )
    train_sample3 = pd.read_pickle(
        "input/amex-pay-features/train_pay_features_part_3.pkl", compression="gzip"
    )
    train_sample4 = pd.read_pickle(
        "input/amex-pay-features/train_pay_features_part_4.pkl", compression="gzip"
    )

    train = pd.concat(
        [train_sample0, train_sample1, train_sample2, train_sample3, train_sample4],
        axis=0,
    )
    del train_sample0, train_sample1, train_sample2, train_sample3, train_sample4

    label = pd.read_csv("input/amex-default-prediction/train_labels.csv")
    train = pd.merge(train, label, on="customer_ID")
    print(train.shape)
    train.to_pickle(
        "input/amex-pay-features/train_pay_features.pkl", compression="gzip"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    args = parser.parse_args()
    main(args)
