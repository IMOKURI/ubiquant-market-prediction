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
from src.features.base import _ALL_FEATURE_NAMES
from src.make_feature import make_feature
from src.time_series_api import TimeSeriesAPI

log = logging.getLogger(__name__)

FEATURES = list(_ALL_FEATURE_NAMES)
FEATURES = [
    "f000",  # f000_initial_features
    "f001",  # f001_last10_features
    "f100",  # f100_vs_same_time_id_average
    "f200",  # f200_lag_last_features
    "f201",  # f201_lag_last_3_features
    "f202",  # f202_lag_last_4_features
    # "f300",  # f300_vs_nearest_neighbors_average
    # "f301",  # f301_nearest_neighbors_target (mean)
    # "f302",  # f302_nearest_neighbors_target_min
    # "f303",  # f303_nearest_neighbors_target_max
    # "f304",  # f304_nearest_neighbors_target_median
    "f400",  # f400_moving_average_short
    "f401",  # ff401_moving_average_long
    "f402",  # f402_moving_average_convergence_divergence
    "f403",  # f403_moving_average_convergence_divergence_signal
    "f901",  # f901_pseudo_last_target
    "f902",  # f902_last10_target
]


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

            pred_df = make_feature(train_df, store, FEATURES, load_from_store=False, save_to_store=False)

            iter_train.predict(sample_prediction_df)

    log.info("Test data")
    with utils.timer("store.empty"):
        store = Store.empty()

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    with utils.timer("store.append"):
        for test_df, sample_prediction_df in iter_test:
            store.update(test_df.values)

            pred_df = make_feature(test_df, store, FEATURES, load_from_store=False, save_to_store=False)

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
