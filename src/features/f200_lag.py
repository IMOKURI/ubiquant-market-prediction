from typing import Dict

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


@feature([f"lag_last_f_{time_id}_{n}" for time_id in range(10) for n in range(300)])
def f210_lag_last_features_last10(ctx: Context) -> Dict[str, float]:
    last_11 = ctx.store.investments[ctx.investment_id].features.last_n(11)
    features = {}
    for time_id, (row_last, row_current) in enumerate(zip(last_11[:-1], last_11[1:])):
        for n, (val_last, val_current) in enumerate(zip(row_last, row_current)):
            features[f"lag_last_f_{time_id - 1}_{n}"] = val_current - val_last
    return features
