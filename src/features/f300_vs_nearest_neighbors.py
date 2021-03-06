from typing import Dict

import numpy as np

from .base import Context, feature
from .helper import *


@feature([f"vs_nn_avg_{n}" for n in range(300)])
def f300_vs_nearest_neighbors_average(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    k = 150

    # scaled = np.empty((1, 300), dtype=np.float32)
    # for n, x in enumerate(latest.T):
    #     x_new = ctx.store.scalers[n].transform(x.reshape(-1, 1))
    #     scaled[0, n] = x_new.squeeze()
    #
    # pca_array = ctx.store.pca.transform(scaled)

    # neigh_index = ctx.store.nearest_neighbors.kneighbors(pca_array, n_neighbors=k, return_distance=False)
    _, nn_index = ctx.store.nearest_neighbors.search(np.ascontiguousarray(latest, dtype=np.float32), k=k)

    avg = np.zeros((300,), dtype=np.float32)
    for index in nn_index[0]:
        avg += ctx.store.training_features[index].squeeze()

    avg /= k

    return {f"vs_nn_avg_{n}": v for n, v in enumerate(latest.squeeze() - avg)}


@feature([f"nn_target_{n}" for n in [3, 9, 27, 81]])
def f301_nearest_neighbors_target(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    _, nn_index = ctx.store.nearest_neighbors.search(np.ascontiguousarray(latest, dtype=np.float32), k=81)

    nn_targets = ctx.store.training_targets[nn_index.squeeze()].squeeze()

    features = {}
    for n in [3, 9, 27, 81]:
        features[f"nn_target_{n}"] = nanmean(nn_targets[:n])

    return features


@feature([f"nn_target_min_{n}" for n in [3, 9, 27, 81]])
def f302_nearest_neighbors_target_min(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    _, nn_index = ctx.store.nearest_neighbors.search(np.ascontiguousarray(latest, dtype=np.float32), k=81)

    nn_targets = ctx.store.training_targets[nn_index.squeeze()].squeeze()

    features = {}
    for n in [3, 9, 27, 81]:
        features[f"nn_target_min_{n}"] = nanmin(nn_targets[:n])

    return features


@feature([f"nn_target_max_{n}" for n in [3, 9, 27, 81]])
def f303_nearest_neighbors_target_max(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    _, nn_index = ctx.store.nearest_neighbors.search(np.ascontiguousarray(latest, dtype=np.float32), k=81)

    nn_targets = ctx.store.training_targets[nn_index.squeeze()].squeeze()

    features = {}
    for n in [3, 9, 27, 81]:
        features[f"nn_target_max_{n}"] = nanmax(nn_targets[:n])

    return features


@feature([f"nn_target_median_{n}" for n in [3, 9, 27, 81]])
def f304_nearest_neighbors_target_median(ctx: Context) -> Dict[str, float]:
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1)

    _, nn_index = ctx.store.nearest_neighbors.search(np.ascontiguousarray(latest, dtype=np.float32), k=81)

    nn_targets = ctx.store.training_targets[nn_index.squeeze()].squeeze()

    features = {}
    for n in [3, 9, 27, 81]:
        features[f"nn_target_median_{n}"] = nanmedian(nn_targets[:n])

    return features
