from typing import Dict

from .base import Context, feature


@feature(["target"])
def f900_target(ctx: Context) -> Dict:
    return {
        "target": ctx.store.investments[ctx.investment_id].targets.last_n(1)[0][0],
    }


@feature(["pseudo_last_target"])
def f901_pseudo_last_target(ctx: Context) -> Dict:
    return {
        "pseudo_last_target": ctx.store.investments[ctx.investment_id].pseudo_targets.last_n(1)[0][0],
    }


@feature([f"target_{n}" for n in range(10)])
def f902_last10_target(ctx: Context) -> Dict:
    return {
        f"target_{n}": v for n, v in enumerate(ctx.store.investments[ctx.investment_id].targets.last_n(10).squeeze())
    }


@feature([f"target_{n}" for n in range(300)])
def f903_feature_as_target(ctx: Context) -> Dict[str, float]:
    return {f"target_{n}": v for n, v in enumerate(ctx.store.investments[ctx.investment_id].features.last_n(1).squeeze())}
