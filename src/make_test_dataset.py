import argparse
import gc

import pandas as pd

from data.dataset import split_dataset
from features.build import add_after_pay_features, build_features


def main(args: argparse.ArgumentParser):
    test = pd.read_parquet(args.path + "test.parquet")
    split_ids = split_dataset(test.customer_ID.unique(), 10)

    for (i, ids) in enumerate(split_ids):
        test_sample = test[test.customer_ID.isin(ids)]
        test_pay_agg = add_after_pay_features(test_sample)
        test_agg = build_features(test_sample)
        test_agg = pd.concat([test_agg, test_pay_agg], axis=1)
        print(i, test_agg.shape)
        test_agg.to_pickle(
            f"input/amex-pay-features/test_pay_features_part_{i}.pkl",
            compression="gzip",
        )
        gc.collect()
        del test_agg, test_pay_agg

    del test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="input/amex-data-parquet/")
    args = parser.parse_args()
    main(args)
