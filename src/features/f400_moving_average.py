from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"ma_short_{n}" for n in range(300)])
def f400_moving_average_short(ctx: Context) -> Dict[str, float]:
    return {
        f"ma_short_{n}": v
        for n, v in enumerate(ctx.store.investments[ctx.investment_id].features_ma_short.last_n(1).squeeze())
    }


@feature([f"ma_long_{n}" for n in range(300)])
def f401_moving_average_long(ctx: Context) -> Dict[str, float]:
    return {
        f"ma_long_{n}": v
        for n, v in enumerate(ctx.store.investments[ctx.investment_id].features_ma_long.last_n(1).squeeze())
    }


@feature([f"macd_{n}" for n in range(300)])
def f402_moving_average_convergence_divergence(ctx: Context) -> Dict[str, float]:
    return {
        f"macd_{n}": v
        for n, v in enumerate(ctx.store.investments[ctx.investment_id].features_macd.last_n(1).squeeze())
    }


@feature([f"macd_signal_{n}" for n in range(300)])
def f403_moving_average_convergence_divergence_signal(ctx: Context) -> Dict[str, float]:
    last_10 = ctx.store.investments[ctx.investment_id].features_macd.last_n(10)
    features = {}
    for n, col in enumerate(last_10.T):
        features[f"macd_signal_{n}"] = np.nanmean(col)
    return features
