import pandas as pd
from typing import Callable, Dict, List, Union

from .feature_store import Store
from .features.base import get_features, get_feature, normalize_feature_name


def get_fingerprint(df):
    head = f"d{len(df)}"
    if len(df) == 3141410:
        head = "train"
    h = hash(frozenset(pd.util.hash_pandas_object(df)))
    hash_value = f"{h:X}"[:6]
    return f"{head}_{hash_value}_{df.loc[0, 'row_id']}_{df.loc[df.index[-1], 'row_id']}"


def make_feature(
        base_df: pd.DataFrame,
        store: Store,
        feature_list: Union[Dict[str, Callable], List[str]] = [],
        with_target: bool = False
):
    fingerprint = get_fingerprint(base_df)

    # 「特徴量の名前」から「特徴量を生成する関数」に変換している
    # TODO: 違う変数名に代入したほうがいいのではないか？
    if feature_list:
        feature_list = {normalize_feature_name(
            k): get_feature(k) for k in feature_list}
    else:
        feature_list = get_features()

    if with_target and "f999_target" not in feature_list:
        feature_list["f999_target"] = get_features()["f999_target"]
