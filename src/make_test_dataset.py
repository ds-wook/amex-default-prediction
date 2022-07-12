import argparse
import gc

import pandas as pd

from data.dataset import split_dataset
from features.build import add_diff_features, build_features


def main(args: argparse.ArgumentParser):
    test = pd.read_parquet(args.path + "test.parquet")
    split_ids = split_dataset(test.customer_ID.unique(), args.num)
    path = "input/amex-rate-features/"

    for (i, ids) in enumerate(split_ids):
        test_sample = test[test.customer_ID.isin(ids)]
        test_agg = add_diff_features(test_sample)
        test_agg = build_features(test_sample)
        print(i, test_agg.shape)

        if args.type == "pkl":
            test_agg.to_pickle(path + args.name + f"_{i}.pkl", compression="gzip")
        else:
            test_agg.to_parquet(path + args.name + f"_{i}.parquet")

        del test_agg
        gc.collect()

    del test
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    parser.add_argument("--name", type=str, default="test_rate_features")
    parser.add_argument("--type", type=str, default="parquet")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()
    main(args)
