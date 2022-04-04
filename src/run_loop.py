import gc
import logging
import os
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import wandb

from .feature_store import Store
from .get_score import get_score
from .make_dataset import make_dataloader, make_dataset, make_dataset_general, make_dataset_lightgbm
from .make_feature import make_feature
from .make_fold import train_test_split
from .make_loss import make_criterion, make_optimizer, make_scheduler
from .make_model import make_model, make_model_tabnet
from .preprocess import apply_faiss_nearest_neighbors, save_training_features, save_training_targets
from .run_epoch import inference_epoch, train_epoch, validate_epoch
from .time_series_api import TimeSeriesAPI
from .utils import AverageMeter, timeSince

# from wandb.lightgbm import log_summary, wandb_callback



log = logging.getLogger(__name__)


def train_fold_lightgbm(c, df, fold):
    train_folds, valid_folds = train_test_split(c, df, fold)
    train_ds, valid_ds = make_dataset_lightgbm(c, train_folds, valid_folds)

    lgb_params = {
        "objective": "regression",
        "boosting": "gbdt",
        "extra_trees": True,  # https://note.com/j26/n/n64d9c37167a6
        "metric": "rmse",
        "learning_rate": 0.05,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.01,
        "lambda_l2": 0.01,
        "num_leaves": 96,
        "max_depth": 12,
        "drop_rate": 0.0,
        "seed": c.params.seed,
    }

    eval_result = {}
    callbacks = [
        lgb.log_evaluation(period=c.settings.print_freq),
        lgb.record_evaluation(eval_result),
        lgb.early_stopping(stopping_rounds=100),
        # wandb_callback(),
    ]

    booster = lgb.train(
        train_set=train_ds,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        params=lgb_params,
        num_boost_round=10000,
        callbacks=callbacks,
    )

    os.makedirs(f"fold{fold}", exist_ok=True)
    joblib.dump(booster, f"fold{fold}/lightgbm.pkl")
    # booster.save_model(f"fold{fold}/lightgbm.pkl", num_iteration=booster.best_iteration)
    # log_summary(booster, save_model_checkpoint=True)

    feature_cols = [f"f_{n}" for n in range(300)]
    valid_folds["preds"] = booster.predict(valid_folds[feature_cols], num_iteration=booster.best_iteration)

    return valid_folds, 0, booster.best_score["valid"]["rmse"]


def train_fold_tabnet(c, df, fold):
    train_folds, valid_folds = train_test_split(c, df, fold)
    train_ds, train_labels, valid_ds, valid_labels = make_dataset_general(c, train_folds, valid_folds)

    clf = make_model_tabnet(c, train_ds)

    clf.fit(
        train_ds,
        train_labels,
        eval_set=[(valid_ds, valid_labels)],
        max_epochs=10000,
        patience=100,
        batch_size=1024 * 20,
        virtual_batch_size=128 * 20,
        num_workers=4,
        drop_last=True,
    )

    os.makedirs(f"fold{fold}", exist_ok=True)
    clf.save_model(f"fold{fold}/tabnet")

    valid_folds["preds"] = clf.predict(valid_ds)

    return valid_folds, 0, clf.best_cost


def train_fold(c, input, fold, device):
    df = input.train
    train_folds, valid_folds = train_test_split(c, df, fold)

    # ====================================================
    # Preprocess
    # ====================================================
    cols = [f"f_{n}" for n in range(300)]
    save_training_features(c, f"training_features_{fold}.npy", train_folds)
    save_training_targets(c, f"training_targets_{fold}.npy", train_folds)
    if "faiss_ivfpq" in c.params.preprocess:
        apply_faiss_nearest_neighbors(c, f"faiss_ivfpq_{fold}.index", train_folds[cols].values)

    # ====================================================
    # Data Loader
    # ====================================================
    # train_ds = make_dataset(c, train_folds)
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
    # scheduler = make_scheduler(c, optimizer, train_ds)
    scheduler = make_scheduler(c, optimizer, df)

    es = EarlyStopping(c=c, fold=fold)

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # ====================================================
        # Training
        # ====================================================
        if c.params.preprocess:
            log.info("Load pretrained features to Store.")
            store = Store.train(c, fold)
        else:
            store = Store.empty()
        iter_train = TimeSeriesAPI(train_folds, scoring=False)
        avg_train_loss = AverageMeter()

        start = time.time()
        for n, (train_df, train_pred_df) in enumerate(iter_train):
            train_df.reset_index(drop=True, inplace=True)
            train_pred_df.reset_index(drop=True, inplace=True)

            gc.collect()

            if c.params.use_feature:
                store.update(train_df.values)

                pred_df = make_feature(
                    train_df,
                    store,
                    c.params.feature_set,
                    c.settings.dirs.feature,
                    with_target=True,
                    fallback_to_none=False,
                    # debug=c.settings.debug,
                )
                train_ds = make_dataset(c, pred_df)

                # assert len(train_df["investment_id"].unique()) == len(
                #     train_df["investment_id"]
                # ), "investment_id is not unique."
                # assert len(train_df) == len(pred_df), "train_df and pred_df do not same size."
            else:
                train_ds = make_dataset(c, train_df)

            train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)

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

            # if c.params.use_feature:
            #     # 推論のときは pseudo label だけど、学習のときは ground truth で。
            #     store.update_post(train_df[["row_id", "target"]].values)

            iter_train.predict(train_pred_df)

            if n % c.settings.print_freq == 0 or n == (len(iter_train) - 1):
                log.info(
                    f"  Training: [{n}/{len(iter_train)}] "
                    f"Elapsed {timeSince(start, float(n + 1) / len(iter_train)):s}, "
                    f"Loss: {avg_train_loss.avg:4f}"
                )

        # ====================================================
        # Validation
        # ====================================================
        if c.params.preprocess:
            log.info("Load pretrained features to Store.")
            store = Store.train(c, fold)
        else:
            store = Store.empty()
        iter_valid = TimeSeriesAPI(valid_folds)
        avg_val_loss = AverageMeter()

        start = time.time()
        for n, (valid_df, valid_pred_df) in enumerate(iter_valid):
            valid_df.reset_index(drop=True, inplace=True)
            valid_pred_df.reset_index(drop=True, inplace=True)

            gc.collect()

            if c.params.use_feature:
                store.update(valid_df.values)

                pred_df = make_feature(
                    valid_df,
                    store,
                    c.params.feature_set,
                    c.settings.dirs.feature,
                    # load_from_store=not c.settings.debug,
                    with_target=True,
                    fallback_to_none=False,
                    # debug=c.settings.debug,
                )
                valid_ds = make_dataset(c, pred_df)

                # assert len(valid_df["investment_id"].unique()) == len(
                #     valid_df["investment_id"]
                # ), "investment_id is not unique."
                # assert len(valid_df) == len(pred_df), "valid_df and pred_df do not same size."
            else:
                valid_ds = make_dataset(c, valid_df)

            valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

            val_loss, preds = validate_epoch(c, valid_loader, model, criterion, device)
            avg_val_loss.update(val_loss, len(valid_df))

            if "LogitsLoss" in c.params.criterion:
                preds = 1 / (1 + np.exp(-preds))

            valid_pred_df[c.params.label_name] = preds

            # if c.params.use_feature:
            #     # 推論のときは pseudo label だけど、学習のときは ground truth で。
            #     store.update_post(valid_df[["row_id", "target"]].values)

            try:
                iter_valid.predict(valid_pred_df)
            except Exception:
                log.error(traceback.format_exc())

            if n % c.settings.print_freq == 0 or n == (len(iter_valid) - 1):
                log.info(
                    f"  Validation: [{n}/{len(iter_valid)}] "
                    f"Elapsed {timeSince(start, float(n + 1) / len(iter_valid)):s}, "
                    f"Loss: {avg_val_loss.avg:4f}"
                )

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

        es(avg_val_loss.avg, iter_valid.score.avg, model, pd.concat(iter_valid.predictions)["target"].values)

        if es.early_stop or os.path.exists(os.path.join(c.settings.dirs.working, "abort-training.flag")):
            log.info("Early stopping")
            break

    if c.params.n_class == 1:
        valid_folds["preds"] = es.best_preds
    elif c.params.n_class > 1:
        valid_folds["preds"] = es.best_preds
        # valid_folds[[str(c) for c in range(c.params.n_class)]] = es.best_preds
        # valid_folds["preds"] = es.best_preds.argmax(1)
    else:
        raise Exception("Invalid n_class.")

    return valid_folds, es.best_score, es.best_loss


def train_fold_batch(c, input, fold, device):
    df = input.train
    train_folds, valid_folds = train_test_split(c, df, fold)

    # ====================================================
    # Data Loader
    # ====================================================
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
    # scheduler = make_scheduler(c, optimizer, train_ds)
    scheduler = make_scheduler(c, optimizer, df)

    es = EarlyStopping(c=c, fold=fold)

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # ====================================================
        # Training
        # ====================================================
        if c.params.skip_training:
            avg_train_loss = 0
        else:
            avg_train_loss = train_epoch(
                c,
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                epoch,
                device,
                verbose=True,
            )

        # ====================================================
        # Validation
        # ====================================================
        avg_val_loss, preds = validate_epoch(c, valid_loader, model, criterion, device, verbose=True)
        valid_labels = valid_folds[c.params.label_name].values

        if "LogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # scoring
        if c.params.n_class == 1 or c.params.model == "ump_ad_ae":
            score = get_score(c.params.scoring, valid_labels, preds)
        elif c.params.n_class > 1:
            score = get_score(c.params.scoring, valid_labels, preds.argmax(1))
        else:
            raise Exception("Invalid n_class.")

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch + 1} - "
            f"train_loss: {avg_train_loss:.4f} "
            f"valid_loss: {avg_val_loss:.4f} "
            f"score: {score:.4f} "
            f"time: {elapsed:.0f}s"
        )
        if c.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"train_loss/fold{fold}": avg_train_loss,
                    f"valid_loss/fold{fold}": avg_val_loss,
                    f"score/fold{fold}": score,
                }
            )

        es(avg_val_loss, score, model, preds)

        if es.early_stop or os.path.exists(os.path.join(c.settings.dirs.working, "abort-training.flag")):
            log.info("Early stopping")
            break

    if c.params.n_class == 1:
        valid_folds["preds"] = es.best_preds
    elif c.params.n_class > 1:
        # valid_folds["preds"] = es.best_preds
        # valid_folds[[str(c) for c in range(c.params.n_class)]] = es.best_preds
        valid_folds["preds"] = 0.0 # es.best_preds.argmax(1)
    else:
        raise Exception("Invalid n_class.")

    return valid_folds, es.best_score, es.best_loss


def inference_lightgbm(df, models):
    predictions = np.zeros((len(df), len(models)), dtype=np.float64)
    feature_cols = [f"f_{n}" for n in range(300)]

    for n, model in enumerate(models):
        preds = model.predict(df[feature_cols].values)
        predictions[:, n] = preds.reshape(-1)

    return predictions


def inference(c, df, device, models):
    predictions = np.zeros((len(df), len(models)), dtype=np.float64)
    # (len(df), len(c.params.pretrained) * c.params.n_fold))

    for n, model in enumerate(models):
        inference_ds = make_dataset(c, df, label=False)
        inference_loader = make_dataloader(c, inference_ds, shuffle=False, drop_last=False)

        # c.params.model = training.model

        # for fold in range(c.params.n_fold):
        # start_time = time.time()

        # model_path = os.path.join(training.dir, f"fold{fold}")
        # model = make_model(c, device, training.dir)

        preds = inference_epoch(c, inference_loader, model, device)

        if "LogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # assert len(df) == len(preds), "Inference result size does not match input size."

        predictions[:, n] = preds

        # elapsed = time.time() - start_time
        # log.info(f"time: {elapsed:.0f}s")

    return predictions


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
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model, ds)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ds):
        """Saves model when validation loss decrease."""
        log.info(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...")
        self.best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(self.dir, self.path))
