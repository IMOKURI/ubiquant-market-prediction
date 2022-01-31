import itertools
import numbers
import sys
from abc import abstractmethod
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     TimeSeriesSplit)


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


# https://github.com/sam31415/timeseriescv/blob/master/timeseriescv/cross_validation.py
class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.
    Time series cross-validation requires each sample has a prediction time pred_time, at which the features are used to
    predict the response, and an evaluation time eval_time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).

    時系列クロスバリデーションのための抽象クラス。
    時系列クロスバリデーションでは、各サンプルに、特徴量を用いて応答を予測する予測時刻 pred_time と、
    応答が判明して誤差を計算できる評価時刻 eval_time が必要です。
    重要なことは、標準的な sklearn クロスバリデーションと異なり、サンプル X、応答 y、pred_times、eval_times は
    すべて同じインデックスを持つ pandas dataframe/series でなければならないということである。
    また、サンプルは予測時間に対して時間順である（すなわち、pred_times は非減少である）ことが仮定されています。

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    """

    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])


# https://zenn.dev/ymd/articles/fd08fb46bc868c
class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation
    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.
    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits folds are used as the test set, while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.
    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.) In addition, an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set. This is to avoid, in the presence of
    temporal correlation, a contamination of the test set by the train set.

    サンプルは、シャッフルせずに、同数のサンプルを含むn_splits fold に分解される。
    各クロス検証ラウンドでは、n_test_splits folds がテストセットとして使用され、他のfoldsはトレーニングセットとして使用される。
    n_splits フォルドのうち、n_test_splits フォルドと同数のラウンドが存在する。 Combinations (10C2 とかのこと)
    各サンプルには、予測時間 pred_time と評価時間 eval_time のタグを付ける。
    分割は，訓練集合とテスト集合のサンプルに関連付けられた区間 [pred_times, eval_times] が重複しないように行う．
    (さらに，テストセットの評価時刻とトレーニングセットの予測時刻の間の最小限の時間を「禁制期間」として定義する．
    これは，時間的相関がある場合に，訓練集合によるテスト集合の汚染を避けるためである．

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.
    time_gap : TODO: need to update
    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).
    """

    def __init__(self, n_splits=10, n_test_splits=2, time_gap=0, embargo_td=0):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits
        if not isinstance(time_gap, int):
            raise ValueError("time gap type is int")
        self.time_gap = time_gap
        if isinstance(embargo_td, int):
            if embargo_td < 0:
                raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        if isinstance(embargo_td, pd.Timedelta):
            if embargo_td < pd.Timedelta(0):
                raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        self.embargo_td = embargo_td

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              pred_times: pd.Series = None,
              eval_times: pd.Series = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.
        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.
        y : pd.Series, not used, inherited from _BaseKFold
        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.
        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        """
        if pred_times is None:
            pred_times = pd.Series(X.index).shift(self.time_gap).bfill()
            pred_times.index = X.index
        if eval_times is None:
            eval_times = pd.Series(X.index).shift(-self.time_gap).ffill()
            eval_times.index = X.index

        super().split(X, y, pred_times, eval_times)

        # Fold boundaries
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_splits)]
        # List of all combinations of n_test_splits folds selected to become test sets
        selected_fold_bounds = list(itertools.combinations(fold_bounds, self.n_test_splits))
        # In order for the first round to have its whole test set at the end of the dataset
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding indices
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices

    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.
        Parameters
        ----------
        test_fold_bounds : List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        """
        # As a first approximation, the train set is the complement of the test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.
        Parameters
        ----------
        fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the test set.
        Returns
        -------
        test_fold_bounds: List of tuples of position indices
            Like fold_bound_list, but with the neighboring folds in the test set merged.
        test_indices: np.ndarray
            A numpy array containing the test indices.
        """
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            # If the current test split is contiguous to the previous one, simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices


def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    """
    Compute a list containing the fold (left) boundaries.
    Parameters
    ----------
    cv: BaseTimeSeriesCrossValidator
        Cross-validation object for which the bounds need to be computed.
    split_by_time: bool
        If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
        time intervals.
    """
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]


def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray, test_indices: np.ndarray,
            test_fold_end: int) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.
    This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
    set sample evaluation times. This method applies the embargo only to the part of the training set immediately
    following the end of the test set determined by test_fold_end.
    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.
    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.
    test_indices : np.ndarray
        A numpy array containing all the indices of the samples in the test set.
    test_fold_end : int
        Index corresponding to the end of a test set block.
    Returns
    -------
    train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.
    """
    if not hasattr(cv, 'embargo_td'):
        raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo"
                         "time.")
    last_test_eval_time = cv.eval_times.iloc[cv.indices[:test_fold_end]].max()
    min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray, test_fold_start: int,
          test_fold_end: int) -> np.ndarray:
    """
    Purge part of the train set.
    Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.
    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.
    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.
    test_fold_start : int
        Index corresponding to the start of a test set block.
    test_fold_end : int
        Index corresponding to the end of the same test set block.
    Returns
    -------
    train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.
    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])

    return np.concatenate((train_indices_1, train_indices_2))
