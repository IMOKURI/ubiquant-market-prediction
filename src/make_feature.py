import logging
import os
from typing import List

import pandas as pd

from .feature_store import Store
from .features.base import Context, get_feature, get_feature_schema, get_features, normalize_feature_name

log = logging.getLogger(__name__)


def get_fingerprint(df):
    head = f"d{len(df)}"
    h = hash(frozenset(pd.util.hash_pandas_object(df)))
    hash_value = f"{h:X}"[:6]
    return f"{head}_{hash_value}_{df.loc[df.index[0], 'row_id']}_{df.loc[df.index[-1], 'row_id']}"


def get_feature_path(base_dir: str, fingerprint: str, fname: str):
    stem = normalize_feature_name(fname)
    return os.path.join(base_dir, fingerprint, f"{stem}.f")


def make_feature(
    base_df: pd.DataFrame,
    store: Store,
    feature_list: List[str] = [],
    feature_store: str = "./features",
    load_from_store: bool = True,
    save_to_store: bool = True,
    with_target: bool = False,
    fallback_to_none: bool = True,
    debug: bool = True,
):
    fingerprint = get_fingerprint(base_df)

    if feature_list:
        feature_dict = {normalize_feature_name(k): get_feature(k) for k in feature_list}
    else:
        feature_dict = get_features()

    if with_target and "f999_target" not in feature_dict:
        feature_dict["f999_target"] = get_features()["f999_target"]

    feature_paths = {fname: get_feature_path(feature_store, fingerprint, fname) for fname in feature_dict}

    if load_from_store:
        feature_list_to_calc = {k: v for k, v in feature_dict.items() if not os.path.exists(feature_paths[k])}
        feature_list_from_cache = {k: v for k, v in feature_dict.items() if k not in feature_list_to_calc}
    else:
        feature_list_to_calc = feature_dict
        feature_list_from_cache = {}

    if not with_target:
        if "f999_target" in feature_list_to_calc:
            del feature_list_to_calc["f999_target"]
        if "f999_target" in feature_list_from_cache:
            del feature_list_from_cache["f999_target"]

    log.debug(
        f"  features to calculate: {list(feature_list_to_calc.keys())}, "
        f"features from cache: {list(feature_list_from_cache.keys())}"
    )

    schema = get_feature_schema()
    feature_to_cols = {}
    dfs = []
    default_feature_cols = [f"f_{n}" for n in range(300)]

    if feature_list_to_calc:
        features = []
        for row in base_df.values:
            feature = {}

            if row.shape[0] == 302:
                investment_id = row[1]
            else:
                investment_id = row[2]

            ctx = Context(base_df[default_feature_cols].values, store, investment_id, fallback_to_none=fallback_to_none)

            for fname, func in feature_list_to_calc.items():

                result = func(ctx)

                for k in result:
                    if k in feature:
                        raise ValueError(f"Feature name {k} is duplicated across features.")

                if fname not in feature_to_cols:
                    feature_to_cols[fname] = list(result.keys())

                if debug:
                    schema_f = schema[fname]
                    for c in feature_to_cols[fname]:
                        assert c in result, f"column schema inconsistent in feature {fname}"
                        assert c in schema_f, f"column schema mismatch, expected: {schema_f}, actual: {c}"

                    for c in schema_f:
                        assert (
                            c in feature_to_cols[fname]
                        ), f"column schema mismatch. {c} not found in generated feature"

                    for c in result.keys():
                        assert c in feature_to_cols[fname], f"column schema inconsistent in feature {fname}"

                feature.update(result)

            features.append(feature)

        features_df = pd.DataFrame(features, dtype="float32")
        dfs.append(features_df)

        if save_to_store:
            os.makedirs(os.path.join(feature_store, fingerprint), exist_ok=True)
            for fname in feature_list_to_calc:
                features_df[feature_to_cols[fname]].to_feather(feature_paths[fname])

    if feature_list_from_cache:
        dfs += [pd.read_feather(feature_paths[fname]) for fname in feature_list_from_cache.keys()]

    assert len(dfs)
    dst = pd.concat(dfs, axis=1)

    # TODO: Nan のときのハンドリング
    assert dst.isnull().values.sum() == 0, f"Feature DataFrame contains Nan. {dst.isnull().values.sum()}"

    return dst
