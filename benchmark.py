import sys

sys.path.append("../inputs")

# isort: split

import cProfile
import logging
from pstats import Stats

import pandas as pd
import ubiquant

import src.utils as utils
from src.feature_store import Store
from src.time_series_api import TimeSeriesAPI

log = logging.getLogger(__name__)


def bench():
    train = pd.read_feather("../inputs/train.f")
    train = train[train["investment_id"] < 100]

    log.info("Training data")
    with utils.timer("store.empty"):
        store = Store.empty()

    log.info(f"Num of train data: {len(train)}")
    iter_train = TimeSeriesAPI(train)

    with utils.timer("store.append"):
        for train_df, sample_prediction_df in iter_train:
            store.update(train_df.values)

            iter_train.predict(sample_prediction_df)

    log.info("Test data")
    with utils.timer("store.empty"):
        store = Store.empty()

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    with utils.timer("store.append"):
        for test_df, sample_prediction_df in iter_test:
            store.update(test_df.values)

            env.predict(sample_prediction_df)


if __name__ == "__main__":
    utils.basic_logger()

    pr = cProfile.Profile()
    pr.enable()

    bench()

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats("tottime").print_stats(50)  # 個別処理時間
    # stats.sort_stats("cumulative").print_stats(50)  # サブ処理含む処理時間
