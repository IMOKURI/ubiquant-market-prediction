import numpy as np
from typing import Dict

from .base import feature, Context


@feature([f"f_{n}" for n in range(300)])
def f000_initial_features(ctx: Context) -> Dict[str, float]:
    return {f"f_{n}": v for n, v in enumerate(ctx.store.investments[ctx.investment_id].features.last_n(1).squeeze())}
