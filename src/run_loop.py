import logging
import os
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import wandb

from .get_score import get_score
from .make_dataset import make_dataloader, make_dataset
from .make_loss import make_criterion, make_optimizer, make_scheduler
from .make_model import make_model
from .run_epoch import train_epoch, validate_epoch, inference_epoch

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
    valid_ds = make_dataset(c, valid_folds)

    train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)
    valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

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

        # train
        if c.params.skip_training:
            avg_loss = 0
        else:
            avg_loss = train_epoch(
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

        # eval
        avg_val_loss, preds = validate_epoch(
            c, valid_loader, model, criterion, device)
        valid_labels = valid_folds[c.params.label_name].values

        if "LogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # scoring
        if c.params.n_class == 1:
            score, prob = get_score(c.params.scoring, valid_labels, preds)
        # elif c.params.n_class > 1:
        #     score = get_score(c.params.scoring, valid_labels, preds.argmax(1))
        else:
            raise Exception("Invalid n_class.")

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch+1} - "
            f"loss_train: {avg_loss:.4f} "
            f"loss_val: {avg_val_loss:.4f} "
            f"score: {score:.4f} "
            f"prob: {prob:.4f} "
            f"time: {elapsed:.0f}s"
        )
        if c.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"loss/train_fold{fold}": avg_loss,
                    f"loss/valid_fold{fold}": avg_val_loss,
                    f"score/fold{fold}": score,
                    f"prob/fold{fold}": prob,
                }
            )

        # es(avg_val_loss, score, model, preds)
        es(avg_val_loss, score, model, preds, train_ds)

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
