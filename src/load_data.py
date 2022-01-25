import logging
import os

import pandas as pd

from .utils import reduce_mem_usage
# from .make_features import create_features, select_features
from .make_fold import make_fold

log = logging.getLogger(__name__)


class InputData:
    def __init__(self, c, use_fold=True):
        self.c = c
        for file_name in c.settings.inputs:
            stem = os.path.splitext(file_name)[0]
            original_file_path = os.path.join(c.settings.dirs.input, file_name)
            f_file_path = original_file_path.replace(".csv", ".f")

            if os.path.exists(f_file_path):
                log.info(f"Load feather file. path: {f_file_path}")
                df = pd.read_feather(f_file_path)

            elif os.path.exists(original_file_path):
                log.info(f"Load original file. path: {original_file_path}")
                df = pd.read_csv(original_file_path)
                df.to_feather(f_file_path)

            else:
                log.warning(f"File does not exist. path: {original_file_path}")
                continue

            # if stem == "train":
            #     create_features(c, df)
            #     df = select_features(c, df)

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            if stem == "train" and use_fold:
                df = make_fold(c, df)

            df = reduce_mem_usage(df)

            setattr(self, stem, df)



class OofData:
    def __init__(self, c, df):
        self.c = c
        self.df = df

        for n, training in enumerate(c.params.pretrained):

            oof_df = pd.read_csv(os.path.join(training.dir, "oof_df.csv"))
            oof_df = oof_df[c.params.label_name]


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data:
        df = df.sample(
            n=c.settings.n_debug_data, random_state=c.params.seed
        ).reset_index(drop=True)
    return df
