import argparse
import gc

import pandas as pd

from data.dataset import split_dataset
from features.build import build_features


def main(args: argparse.ArgumentParser):
    test = pd.read_parquet(args.path + "test.parquet")
    split_ids = split_dataset(test.customer_ID.unique(), 10)
    path = "input/amex-trick-features/"

    for (i, ids) in enumerate(split_ids):
        test_sample = test[test.customer_ID.isin(ids)]

        test_agg = build_features(test_sample)
        print(i, test_agg.shape)
        test_agg.to_pickle(path + args.name + f"_{i}.pkl", compression="gzip")
        del test_agg
        gc.collect()

    del test
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    parser.add_argument("--name", type=str, default="test_diff_features")
    args = parser.parse_args()
    main(args)
