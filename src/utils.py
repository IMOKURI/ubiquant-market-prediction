import json
import logging
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Optional

import git
import numpy as np
import pandas as pd
import requests
import torch
import wandb
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

log = logging.getLogger(__name__)

_ALWAYS_CATCH = False


def basic_logger():
    logging.basicConfig(
        # filename=__file__.replace('.py', '.log'),
        level=logging.getLevelName("INFO"),
        format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
    )


def set_always_catch(catch: bool):
    global _ALWAYS_CATCH
    _ALWAYS_CATCH = catch


def in_kaggle() -> bool:
    return 'kaggle_web_client' in sys.modules


@contextmanager
def catch_everything_in_kaggle(name: Optional[str] = None):
    try:
        yield
    except Exception:
        msg = f"WARNINGS: exception occurred in {name or '(unknown)'}: {traceback.format_exc()}"
        log.warning(msg)
        print(msg)

        if in_kaggle() or _ALWAYS_CATCH:
            # ...catch and suppress if this is executed in kaggle
            pass
        else:
            # re-raise if this is executed outside of kaggle
            raise


def fix_seed(seed=42):
    log.info(f"Fix seed: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def debug_settings(c):
    if c.settings.debug:
        log.info("Enable debug mode.")
        c.wandb.enabled = False
        c.settings.print_freq = 10
        c.params.n_fold = 3
        c.params.epoch = 1


def gpu_settings(c):
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = c.settings.gpus
        log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    except ConfigAttributeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"torch device: {device}, device count: {torch.cuda.device_count()}")
    return device


def df_stats(df):
    stats = []
    for col in df.columns:
        try:
            stats.append(
                (
                    col,
                    df[col].nunique(),
                    df[col].value_counts().index[0],
                    df[col].value_counts().values[0],
                    df[col].value_counts(
                        normalize=True, dropna=False).values[0] * 100,
                    df[col].isnull().sum() * 100 / df.shape[0],
                    df[col].dtype,
                )
            )
        except TypeError:
            log.warning(f"Skip column. {col}: {df[col].dtype}")
    return pd.DataFrame(
        stats, columns=["カラム名", "ユニーク値数", "最頻値",
                        "最頻値の出現回数", "最頻値の割合", "欠損値の割合", "タイプ"]
    )


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024 ** 2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        log.info(
            f"Mem. usage decreased to {str(end_mem)[:3]}Mb: {num_reduction[:2]}% reduction")

    return df_out


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


@contextmanager
def timer(name):
    s = time.time()
    yield
    elapsed = time.time() - s
    log.info(f"[{name}] {asMinutes(elapsed)} ({elapsed:.3f}s)")


def compute_grad_norm(parameters, norm_type=2.0):
    """Refer to torch.nn.utils.clip_grad_norm_"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device)
             for p in parameters]
        ),
        norm_type,
    )

    return total_norm


def setup_wandb(c):
    if c.wandb.enabled:
        os.makedirs(os.path.abspath(c.wandb.dir), exist_ok=True)
        c_dict = OmegaConf.to_container(c.params, resolve=True)
        c_dict["commit"] = get_commit_hash(c.settings.dirs.working)
        run = wandb.init(
            entity=c.wandb.entity,
            project=c.wandb.project,
            dir=os.path.abspath(c.wandb.dir),
            config=c_dict,
            group=c.wandb.group,
        )
        log.info(f"WandB initialized. name: {run.name}, id: {run.id}")
        return run


def teardown_wandb(c, run, loss):
    if c.wandb.enabled:
        wandb.summary["loss"] = loss
        artifact = wandb.Artifact(
            # c.params.model_name.replace("/", "-"), type="model")
            c.params.model.replace("/", "-"), type="model")
        artifact.add_dir(".")
        run.log_artifact(artifact)
        log.info(f"WandB recorded. name: {run.name}, id: {run.id}")


def get_commit_hash(dir_):
    repo = git.Repo(dir_, search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def send_result_to_slack(score, loss):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    msg = f"Run at: {os.path.basename(os.getcwd())}, score: {score:.5f}, loss: {loss:.5f}"
    try:
        requests.post(webhook_url, data=json.dumps({"text": msg}))
    except Exception:
        log.warning("Failed to send message to slack.")
