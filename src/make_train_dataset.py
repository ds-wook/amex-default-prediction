import argparse

import pandas as pd

from features.build import add_after_pay_features, build_features


def main(args: argparse.ArgumentParser):
    train = pd.read_parquet(args.path + "train.parquet")
    after_pay = add_after_pay_features(train)
    train = build_features(train)
    train = pd.concat([train, after_pay], axis=1)
    del after_pay
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
