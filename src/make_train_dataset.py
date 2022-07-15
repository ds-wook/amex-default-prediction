import argparse
import gc

import pandas as pd

from data.dataset import split_dataset
from features.build import add_diff_features, build_features


def main(args: argparse.ArgumentParser):
    train = pd.read_parquet(args.path + "train.parquet")
    split_ids = split_dataset(train.customer_ID.unique(), args.num)
    path = "input/amex-trick-features/"

    for (i, ids) in enumerate(split_ids):
        train_sample = train[train.customer_ID.isin(ids)]
        train_agg = add_diff_features(train_sample)
        train_agg = build_features(train_sample)

        print(i, train_agg.shape)

        if args.type == "pkl":
            train_agg.to_pickle(path + args.name + f"_{i}.pkl", compression="gzip")

        else:
            train_agg.to_parquet(path + args.name + f"_{i}.parquet")

        del train_agg
        gc.collect()

    if args.type == "pkl":
        train = pd.read_pickle(path + args.name + "_0.pkl", compression="gzip")

        for num in range(1, args.num):
            train_sample = pd.read_pickle(
                path + args.name + f"_{num}.pkl", compression="gzip"
            )
            train = pd.concat([train, train_sample], axis=0)
            del train_sample
            gc.collect()

        label = pd.read_csv("input/amex-default-prediction/train_labels.csv")
        train = pd.merge(train, label, on="customer_ID")
        print(train.shape)
        train.to_pickle(path + args.name + ".pkl", compression="gzip")

    else:
        train = pd.read_parquet(path + args.name + "_0.parquet")

        for num in range(1, args.num):
            train_sample = pd.read_parquet(path + args.name + f"_{num}.parquet")
            train = pd.concat([train, train_sample], axis=0)
            del train_sample
            gc.collect()

        label = pd.read_csv("input/amex-default-prediction/train_labels.csv")
        train = pd.merge(train, label, on="customer_ID")
        print(train.shape)
        train.to_parquet(path + args.name + ".parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    parser.add_argument("--name", type=str, default="train_trick_features")
    parser.add_argument("--type", type=str, default="parquet")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()
    main(args)
