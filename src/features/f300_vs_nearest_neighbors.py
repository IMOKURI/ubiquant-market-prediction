from typing import Dict

import numpy as np

from .base import Context, feature


@feature([f"vs_nn_avg_{n}" for n in range(300)])
def f300_vs_nearest_neighbors_average(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    scaled = np.empty((1, 300), dtype=np.float32)
    for n, x in enumerate(latest.T):
        x_new = ctx.store.scalers[n].transform(x.reshape(-1, 1))
        scaled[0, n] = x_new.squeeze()

    pca_array = ctx.store.pca.transform(scaled)

    neigh_index = ctx.store.nearest_neighbors.kneighbors(pca_array, n_neighbors=10, return_distance=False)

    avg = np.zeros((300,), dtype=np.float32)
    for index in neigh_index[0]:
        avg += ctx.store.pca.inverse_transform(ctx.store.sampling_array[index].reshape(1, -1)).squeeze()

    avg /= 10

    return {f"vs_nn_avg_{n}": v for n, v in enumerate(latest.squeeze() - avg)}
