from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"lag_last_f_{n}" for n in range(300)])
def f200_lag_last_features(ctx: Context) -> Dict[str, float]:
    last_n = ctx.store.investments[ctx.investment_id].features.last_n(2)
    last_n_lag = last_n[-1] - last_n[0]
    return {f"lag_last_f_{n}": v for n, v in enumerate(last_n_lag.squeeze())}


@feature([f"lag_last_3_f_{n}" for n in range(300)])
def f201_lag_last_3_features(ctx: Context) -> Dict[str, float]:
    last_n = ctx.store.investments[ctx.investment_id].features.last_n(3)
    last_n_lag = last_n[-1] - last_n[0]
    return {f"lag_last_3_f_{n}": v for n, v in enumerate(last_n_lag.squeeze())}


@feature([f"lag_last_4_f_{n}" for n in range(300)])
def f202_lag_last_4_features(ctx: Context) -> Dict[str, float]:
    last_n = ctx.store.investments[ctx.investment_id].features.last_n(4)
    last_n_lag = last_n[-1] - last_n[0]
    return {f"lag_last_4_f_{n}": v for n, v in enumerate(last_n_lag.squeeze())}
