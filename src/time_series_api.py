import logging
import warnings

import numpy as np
import scipy.stats as ss

from .utils import AverageMeter

log = logging.getLogger(__name__)


class TimeSeriesAPI:
    def __init__(self, df, scoring=True):
        df = df.reset_index(drop=True)
        self.df = df
        self.target = df["target"].values

        df_groupby_timeid = df.groupby("time_id")
        self.df_iter = df_groupby_timeid.__iter__()
        self.init_num_timeid = len(df_groupby_timeid)

        self.next_calls = 0
        self.pred_calls = 0

        self.predictions = []
        self.targets = []

        self.scoring = scoring
        self.score = AverageMeter()
        self.rs = []
        self.probs = []

    def __iter__(self):
        return self

    def __len__(self):
        return self.init_num_timeid  # - self.next_calls

    def __next__(self):
        assert self.pred_calls == self.next_calls, "You must call `predict()` before you get the next batch of data."

        time_id, df = next(self.df_iter)
        self.next_calls += 1

        data_df = df  # .drop(columns=["time_id", "target"])

        target_df = df[["row_id", "target", "investment_id"]]
        self.targets.append(target_df)

        pred_df = target_df.drop(columns=["investment_id"])
        pred_df["target"] = 0.0

        return data_df, pred_df

    def predict(self, pred_df):
        assert self.pred_calls == self.next_calls - 1, "You must get the next batch before making a new prediction."
        assert pred_df.columns.to_list() == ["row_id", "target"], "Prediction dataframe have invalid columns."

        pred_df = pred_df.astype({"row_id": np.dtype("str"), "target": np.dtype("float32")})

        self.predictions.append(pred_df)
        self.pred_calls += 1

        if pred_df["target"].isnull().sum() > 0:
            log.warning(f"The prediction contains Nan.\n{pred_df[pred_df['target'].isnull()]}")

        if self.scoring:
            try:
                assert len(pred_df) > 1
            except Exception:
                log.warning("Prediction DataFrame contains only one investment_id.")
                return

            try:
                # r: ピアソン相関係数 -1 <= r <= 1 の値を取り、1に近いほど正の相関、-1に近いほど負の相関、0に近いほど無相関
                # prob: 相関が優位かどうかを示す。例えば、 p = 0.05 なら実際には相関がないにも関わらず、相関が存在すると
                #       判定されるリスクが 5% ある。概ね、 p <= 0.05 なら相関が 0 とは異なる（優位である）と結論付けられる。
                r, prob = ss.pearsonr(pred_df["target"].values, self.targets[-1]["target"].values)
                # log.debug(f"pred_df: {pred_df['target'].values}")

                self.score.update(r)
                self.rs.append(r)
                self.probs.append(prob)
            except ValueError as e:
                log.warning(f"TimeSeriesAPI.predict: {e}")
