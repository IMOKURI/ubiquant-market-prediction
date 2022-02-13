from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"vs_nn_avg_{n}" for n in range(300)])
def f300_vs_nearest_neighbors_average(ctx: Context) -> Dict[str, float]:
    mean = np.nanmean(ctx.base_array, axis=0)
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1).squeeze()
    return {f"vs_nn_avg_{n}": v for n, v in enumerate(latest - mean)}
