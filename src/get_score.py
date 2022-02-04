import logging
import traceback

import numpy as np
import scipy.stats as stats
import wandb
from sklearn.metrics import accuracy_score, mean_squared_error

log = logging.getLogger(__name__)


def get_score(scoring, y_true, y_pred):
    if scoring == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif scoring == "pearson":
        try:
            return stats.pearsonr(y_true, y_pred)
        except ValueError:
            log.warning(traceback.format_exc())
            return None, None

    else:
        raise Exception("Invalid scoring.")


def record_result(c, df, fold, loss=None):
    prob = 1.0
    if c.params.scoring == "mean":
        score = df["preds"].mean()
    elif c.params.scoring == "pearson":
        preds = df["preds"].values
        labels = df[c.params.label_name].values
        score, prob = get_score(c.params.scoring, labels, preds)
    else:
        preds = df["preds"].values
        labels = df[c.params.label_name].values
        score = get_score(c.params.scoring, labels, preds)

    log.info(f"Score: {score:<.5f}")
    if c.wandb.enabled:
        wandb.log({"score": score, "prob": prob, "fold": fold})
        if loss is not None:
            wandb.log({"loss": loss, "fold": fold})

    return score
