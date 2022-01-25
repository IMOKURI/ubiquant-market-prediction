import logging
import os
import sys

import hydra

import src.utils as utils
from src.run_loop import inference

sys.path.append("../inputs")

import ubiquant


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info(f"Started at {os.path.basename(os.getcwd())}")

    utils.fix_seed(c.params.seed)
    device = utils.gpu_settings(c)

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    for n, (test_df, sample_prediction_df) in enumerate(iter_test):

        df = inference(c, test_df, device)

        sample_prediction_df[c.params.label_name] = df[c.params.label_name]
        env.predict(sample_prediction_df)


if __name__ == "__main__":
    main()
