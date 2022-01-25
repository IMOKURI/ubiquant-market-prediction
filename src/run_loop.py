import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import wandb

from .utils import AverageMeter
# from .get_score import get_score
from .make_dataset import make_dataloader, make_dataset
from .make_loss import make_criterion, make_optimizer, make_scheduler
from .make_model import make_model
from .run_epoch import train_epoch, validate_epoch, inference_epoch
from .time_series_api import TimeSeriesAPI

log = logging.getLogger(__name__)


def train_fold(c, df, fold, device):
    # ====================================================
    # Data Loader
    # ====================================================
    trn_idx = df[df["fold"] != fold].index
    val_idx = df[df["fold"] == fold].index
    log.info(
        f"Num of training data: {len(trn_idx)}, num of validation data: {len(val_idx)}")

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)

    train_ds = make_dataset(c, train_folds)
    # valid_ds = make_dataset(c, valid_folds)

    # train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)
    # valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

    # ====================================================
    # Model
    # ====================================================
    model = make_model(c, device)

    criterion = make_criterion(c)
    optimizer = make_optimizer(c, model)
    scaler = amp.GradScaler(enabled=c.settings.amp)
    scheduler = make_scheduler(c, optimizer, train_ds)

    es = EarlyStopping(c=c, fold=fold)

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # ====================================================
        # Training
        # ====================================================
        iter_train = TimeSeriesAPI(train_folds)
        avg_train_loss = AverageMeter()

        for train_df, train_pred_df in iter_train:
            train_ds = make_dataset(c, train_df)
            train_loader = make_dataloader(
                c, train_ds, shuffle=True, drop_last=True)

            if c.params.skip_training:
                train_loss = 0
            else:
                train_loss = train_epoch(
                    c,
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    device,
                )
            avg_train_loss.update(train_loss, len(train_df))

            iter_train.predict(train_pred_df)

        # ====================================================
        # Validation
        # ====================================================
        iter_valid = TimeSeriesAPI(valid_folds)
        avg_val_loss = AverageMeter()

        for valid_df, valid_pred_df in iter_valid:
            valid_ds = make_dataset(c, valid_df)
            valid_loader = make_dataloader(
                c, valid_ds, shuffle=False, drop_last=False)

            val_loss, preds = validate_epoch(
                c, valid_loader, model, criterion, device)
            avg_val_loss.update(val_loss, len(valid_df))

            if "LogitsLoss" in c.params.criterion:
                preds = 1 / (1 + np.exp(-preds))

            valid_pred_df[c.params.label_name] = preds
            iter_valid.predict(valid_pred_df)

        # scoring
        # if c.params.n_class == 1:
        #     score, prob = get_score(c.params.scoring, valid_labels, preds)
        # elif c.params.n_class > 1:
        #     score = get_score(c.params.scoring, valid_labels, preds.argmax(1))
        # else:
        #     raise Exception("Invalid n_class.")

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch+1} - "
            f"train_loss: {avg_train_loss.avg:.4f} "
            f"valid_loss: {avg_val_loss.avg:.4f} "
            f"score: {iter_valid.score.avg:.4f} "
            f"prob: {np.mean(iter_valid.probs):.4f} "
            f"time: {elapsed:.0f}s"
        )
        if c.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"train_loss/fold{fold}": avg_train_loss.avg,
                    f"valid_loss/fold{fold}": avg_val_loss.avg,
                    f"score/fold{fold}": iter_valid.score.avg,
                    f"prob/fold{fold}": np.mean(iter_valid.probs),
                }
            )

        es(avg_val_loss.avg, iter_valid.score.avg,
           model, pd.concat(iter_valid.predictions))

        if es.early_stop:
            log.info("Early stopping")
            break

    if c.params.n_class == 1:
        valid_folds["preds"] = es.best_preds
    # elif c.params.n_class > 1:
    #     valid_folds[[str(c) for c in range(c.params.n_class)]] = es.best_preds
    #     valid_folds["preds"] = es.best_preds.argmax(1)
    else:
        raise Exception("Invalid n_class.")

    return valid_folds, es.best_score, es.best_loss


def inference(c, df, device):
    predictions = np.zeros(
        (len(df), len(c.params.pretrained) * c.params.n_fold))
    n = 0

    for training in c.params.pretrained:
        inference_ds = make_dataset(c, df, label=False)
        inference_loader = make_dataloader(
            c, inference_ds, shuffle=False, drop_last=False)

        c.params.model = training.model
        c.params.model_name = training.name

        for fold in range(c.params.n_fold):
            start_time = time.time()

            model_path = os.path.join(training.dir, f"fold{fold}")
            model = make_model(c, device, model_path)

            preds = inference_epoch(c, inference_loader, model, device)

            if "LogitsLoss" in c.params.criterion:
                preds = 1 / (1 + np.exp(-preds))

            predictions[:, n] = preds
            n += 1

            elapsed = time.time() - start_time
            log.info(f"time: {elapsed:.0f}s")

    df[c.params.label_name] = np.mean(predictions, axis=1)

    return df


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, c, fold, delta=0):
        self.patience = c.params.es_patience
        self.dir = f"fold{fold}"
        self.path = "pytorch_model.bin"
        os.makedirs(self.dir, exist_ok=True)

        self.early_stop = False
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.best_loss = np.Inf
        self.best_preds = None

    def __call__(self, val_loss, score, model, preds, ds=None):

        if self.best_score is None:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model, ds)
        elif val_loss >= self.best_loss + self.delta:
            if self.patience <= 0:
                return
            self.counter += 1
            log.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model, ds)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ds):
        """Saves model when validation loss decrease."""
        log.info(
            f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ..."
        )
        self.best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(self.dir, self.path))
