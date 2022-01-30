import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit


def make_fold(c, df):
    if c.params.fold == "bins_stratified":
        df = bins_stratified_kfold(c, df, c.params.label_name)
    elif c.params.fold == "stratified":
        df = stratified_kfold(c, df, c.params.label_name)
    elif c.params.fold == "group":
        df = group_kfold(c, df, c.params.group_name)
    elif c.params.fold == "time_series":
        df = moving_window_kfold(c, df, c.params.time_name)
    elif c.params.fold == "time_series_group":
        df = moving_window_group_kfold(c, df, c.params.group_name, c.params.time_name)

    else:
        raise Exception("Invalid fold.")

    return df


def bins_stratified_kfold(c, df, col):
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[col], bins=num_bins, labels=False)

    fold_ = StratifiedKFold(n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df["bins"])):
        df.loc[val_index, "fold"] = int(n)

    return df


def stratified_kfold(c, df, col):
    fold_ = StratifiedKFold(n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


def group_kfold(c, df, col):
    fold_ = GroupKFold(n_splits=c.params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df, groups=df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


# def time_series_split(c, df):
#     fold_ = TimeSeriesSplit(n_splits=c.params.n_fold)
#     for n, (_, val_index) in enumerate(fold_.split(df)):
#         df.loc[val_index, "fold"] = int(n)
#
#     return df


def moving_window_kfold(c, df, col):
    fold_ = MovingWindowKFold(col, clipping=False, n_splits=c.params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df)):
        df.loc[df.index[val_index], "fold"] = int(n)

    return df


def moving_window_group_kfold(c, df, group_col, time_col):
    fold_ = GroupKFold(n_splits=c.params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df, groups=df[group_col])):
        df.loc[val_index, "group_fold"] = int(n)

    fold_ = MovingWindowKFold(time_col, clipping=False, n_splits=c.params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df)):
        df.loc[df.index[val_index], "time_fold"] = int(n)

    return df


# https://blog.amedama.jp/entry/time-series-cv
class MovingWindowKFold(TimeSeriesSplit):
    """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

    def __init__(self, ts_column, clipping=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.ts_column = ts_column
        # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
        self.clipping = clipping

    def split(self, X, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(X, pd.DataFrame)

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        ts = X[self.ts_column]
        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        ts_df = ts.reset_index()
        # 時系列でソートする
        sorted_ts_df = ts_df.sort_values(by=self.ts_column)
        # スーパークラスのメソッドで添字を計算する
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # 添字を元々の DataFrame の iloc として使える値に変換する
            train_iloc_index = sorted_ts_df.iloc[train_index].index
            test_iloc_index = sorted_ts_df.iloc[test_index].index

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])
