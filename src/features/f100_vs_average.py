from typing import Dict

from .base import Context, feature
from .helper import *


@feature([f"vs_same_time_avg_{n}" for n in range(300)])
def f100_vs_same_time_id_average(ctx: Context) -> Dict[str, float]:
    mean = nanmean(ctx.base_array, axis=0)
    latest = ctx.store.investments[ctx.investment_id].features.last_n(1).squeeze()
    return {f"vs_same_time_avg_{n}": v for n, v in enumerate(latest - mean)}
