from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"vs_nn_avg_{n}" for n in range(300)])
def f300_vs_nearest_neighbors_average(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)
    neigh_index = ctx.store.nearest_neighbors.kneighbors(latest, n_neighbors=11, return_distance=False)

    avg = np.zeros((300,), dtype=np.float32)
    for index in neigh_index[0][1:]:  # 1つ目のindexは自分自身なので除外する
        avg += ctx.store.training_array[index]

    avg /= 10

    return {f"vs_nn_avg_{n}": v for n, v in enumerate(latest.squeeze() - avg)}
