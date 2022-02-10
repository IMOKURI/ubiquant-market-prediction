from typing import Dict

from .base import Context, feature


@feature(["target"])
def f900_target(ctx: Context) -> Dict:
    return {
        "target": ctx.store.investments[ctx.investment_id].targets.last_n(1)[0][0],
    }


@feature(["pseudo_target"])
def f901_pseudo_target(ctx: Context) -> Dict:
    return {
        "pseudo_target": ctx.store.investments[ctx.investment_id].pseudo_targets.last_n(1)[0][0],
    }
