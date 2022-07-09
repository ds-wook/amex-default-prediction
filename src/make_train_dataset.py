import argparse
import gc

import pandas as pd

from data.dataset import split_dataset
from features.build import build_features


def main(args: argparse.ArgumentParser):
    train = pd.read_parquet(args.path + "train.parquet")
    split_ids = split_dataset(train.customer_ID.unique(), 5)
    path = "input/amex-trick-features/"

    for (i, ids) in enumerate(split_ids):
        train_sample = train[train.customer_ID.isin(ids)]
        train_agg = build_features(train_sample)

        print(i, train_agg.shape)

        train_agg.to_pickle(path + args.name + f"_{i}.pkl", compression="gzip")
        del train_agg
        gc.collect()

    train_sample0 = pd.read_pickle(path + args.name + "_0.pkl", compression="gzip")
    train_sample1 = pd.read_pickle(path + args.name + "_1.pkl", compression="gzip")
    train_sample2 = pd.read_pickle(path + args.name + "_2.pkl", compression="gzip")
    train_sample3 = pd.read_pickle(path + args.name + "_3.pkl", compression="gzip")
    train_sample4 = pd.read_pickle(path + args.name + "_4.pkl", compression="gzip")

    train = pd.concat(
        [train_sample0, train_sample1, train_sample2, train_sample3, train_sample4],
        axis=0,
    )
    del train_sample0, train_sample1, train_sample2, train_sample3, train_sample4
    gc.collect()

    label = pd.read_csv("input/amex-default-prediction/train_labels.csv")
    train = pd.merge(train, label, how="inner", on="customer_ID")
    print(train.shape)
    train.to_pickle(path + args.name + ".pkl", compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    parser.add_argument("--name", type=str, default="train_diff_features")
    args = parser.parse_args()
    main(args)
