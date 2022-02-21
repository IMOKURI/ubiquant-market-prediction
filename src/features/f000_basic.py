from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"f_{n}" for n in range(300)])
def f000_initial_features(ctx: Context) -> Dict[str, float]:
    return {f"f_{n}": v for n, v in enumerate(ctx.store.investments[ctx.investment_id].features.last_n(1).squeeze())}


@feature([f"f_{time_id}_{n}" for time_id in range(10) for n in range(300)])
def f001_last10_features(ctx: Context) -> Dict[str, float]:
    last_10 = ctx.store.investments[ctx.investment_id].features.last_n(10)
    features = {}
    for time_id, row in enumerate(last_10):
        for n, val in enumerate(row):
            features[f"f_{time_id}_{n}"] = val
    return features
