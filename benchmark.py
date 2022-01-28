import sys

sys.path.append("../inputs")

import ubiquant

import src.utils as utils
from src.feature_store import Store


def bench():
    with utils.timer("store.empty"):
        store = Store.empty()

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    for test_df, sample_prediction_df in iter_test:
        with utils.timer("store.append"):
            for _, row in test_df.iterrows():
                store.append(row)

        env.predict(sample_prediction_df)


if __name__ == "__main__":
    utils.basic_logger()
    bench()
