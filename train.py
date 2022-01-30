import logging
import os

import hydra
import pandas as pd
from omegaconf.errors import ConfigAttributeError

import src.utils as utils
from src.get_score import record_result
from src.load_data import InputData
from src.run_loop import train_fold

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    utils.debug_settings(c)
    run = utils.setup_wandb(c)

    log.info(f"Started at {os.path.basename(os.getcwd())}")

    utils.fix_seed(c.params.seed)
    device = utils.gpu_settings(c)

    input = InputData(c)

    oof_df = pd.DataFrame()
    losses = utils.AverageMeter()

    single_run = False

    for fold in range(c.params.n_fold):
        try:
            fold = int(c.settings.run_fold)
            single_run = True
        except ConfigAttributeError:
            pass

        log.info(f"========== fold {fold} training ==========")
        utils.fix_seed(c.params.seed + fold)

        _oof_df, score, loss = train_fold(c, input.train, fold, device)
        oof_df = pd.concat([oof_df, _oof_df])
        losses.update(loss)

        log.info(f"========== fold {fold} result ==========")
        record_result(c, _oof_df, fold, loss)

        if c.settings.debug or single_run:
            break

    # oof_df.to_csv("oof_df.csv", index=False)
    if c.params.fold == "time_series_group":
        oof_df[["row_id", "time_id", "investment_id", "target", "preds", "group_fold"]].to_feather("oof_df.f")
    else:
        oof_df[["row_id", "time_id", "investment_id", "target", "preds", "fold"]].to_feather("oof_df.f")

    log.info("========== final result ==========")
    score = record_result(c, oof_df, c.params.n_fold, losses.avg)

    log.info("Done.")

    utils.teardown_wandb(c, run, losses.avg)
    utils.send_result_to_slack(score, losses.avg)


if __name__ == "__main__":
    main()
