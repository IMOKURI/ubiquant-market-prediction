from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"lag_last_f_{n}" for n in range(300)])
def f200_lag_last_features(ctx: Context) -> Dict[str, float]:
    last_2 = ctx.store.investments[ctx.investment_id].features.last_n(2)
    last_2_lag = last_2[-1] - last_2[-2]
    return {f"lag_last_f_{n}": v for n, v in enumerate(last_2_lag.squeeze())}
