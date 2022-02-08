import logging
import time
import warnings

import numpy as np
import torch
import torch.cuda.amp as amp

from .utils import AverageMeter, compute_grad_norm, timeSince

log = logging.getLogger(__name__)


def train_epoch(c, train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device):
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    # start = time.time()

    for step, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with amp.autocast(enabled=c.settings.amp):
            # y_preds = model(images)
            # y_preds = model(images).squeeze()
            y_preds = model(features).squeeze()

            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)
            loss = loss / c.params.gradient_acc_step

        scaler.scale(loss).backward()

        if (step + 1) % c.params.gradient_acc_step == 0:
            scaler.unscale_(optimizer)

            # error_if_nonfinite に関する warning を抑止する
            # pytorch==1.10 で不要となりそう
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", FutureWarning)
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         model.parameters(), c.params.max_grad_norm  # , error_if_nonfinite=False
            #     )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            ...
            # grad_norm = compute_grad_norm(model.parameters())

        # if c.params.scheduler == "CosineAnnealingWarmupRestarts":
        #     last_lr = scheduler.get_lr()[0]
        # else:
        #     last_lr = scheduler.get_last_lr()[0]

        # end = time.time()
        # if step % c.settings.print_freq == 0 or step == (len(train_loader) - 1):
        #     log.info(
        #         f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
        #         f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
        #         f"Loss: {losses.avg:.4f} "
        #         f"Grad: {grad_norm:.4f} "
        #         f"LR: {last_lr:.2e}  "
        #     )

    return losses.avg


def validate_epoch(c, valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()

    size = len(valid_loader.dataset)
    preds = np.zeros((size,))
    # start = time.time()

    for step, (features, labels) in enumerate(valid_loader):
        features = features.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # with torch.inference_mode():
        with torch.no_grad():
            # y_preds = model(images)
            # y_preds = model(images).squeeze()
            y_preds = model(features).squeeze()

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        begin = step * c.params.batch_size
        end = begin + batch_size
        if c.params.n_class == 1:
            preds[begin:end] = y_preds.to("cpu").numpy()
        # elif c.params.n_class > 1:
        #     preds[begin:end] = y_preds.softmax(1).to("cpu").numpy()
        else:
            raise Exception("Invalid n_class.")

        # end = time.time()
        # if step % c.settings.print_freq == 0 or step == (len(valid_loader) - 1):
        #     log.info(
        #         f"EVAL: [{step}/{len(valid_loader)}] "
        #         f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
        #         f"Loss: {losses.avg:.4f} "
        #     )

    return losses.avg, preds


def inference_epoch(c, inference_loader, model, device):
    model.eval()

    size = len(inference_loader.dataset)
    preds = np.zeros((size,), dtype=np.float64)
    # start = time.time()

    for step, features in enumerate(inference_loader):
        features = features.to(device)
        batch_size = features.size(0)

        # with torch.inference_mode():
        with torch.no_grad():
            # y_preds = model(images)
            # y_preds = model(images).squeeze()
            y_preds = model(features).squeeze()

        begin = step * c.params.batch_size
        end = begin + batch_size
        if c.params.n_class == 1:
            preds[begin:end] = y_preds.to("cpu").numpy()
        # elif c.params.n_class > 1:
        #     preds[begin:end] = y_preds.softmax(1).to("cpu").numpy()
        else:
            raise Exception("Invalid n_class.")

        # end = time.time()
        # if step % c.settings.print_freq == 0 or step == (len(inference_loader) - 1):
        #     log.info(
        #         f"EVAL: [{step}/{len(inference_loader)}] "
        #         f"Elapsed {timeSince(start, float(step + 1) / len(inference_loader)):s} "
        #     )

    return preds
