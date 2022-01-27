import numpy as np
import pandas as pd
from typing import Dict

from .streamdf import StreamDf


class Investment:
    def __init__(self, row: pd.Seies, investment_id: int, features: StreamDf):
        self.row = row
        self.investment_id = investment_id
        self.features = features

    @classmethod
    def empty(cls, row: pd.Series, investment_id):
        feature_schema = {'time_id': np.int32}

        feature_cols = [f"f_{n}" for n in range(300)]
        for col in feature_cols:
            feature_schema[col] = np.float32

        features = StreamDf.empty(feature_schema, 'time_id')

        return Investment(row, investment_id, features)

    def last_n(self, n: int) -> 'Investment':
        return Investment(self.row, self.investment_id, self.features.last_n(n))


class Investments:
    def __init__(self, investment_df: pd.DataFrame):
        self.investment_df = investment_df.set_index("investment_id")
        self.investments = {}  # type: Dict[int, Investment]
        self.feature_cols = ["time_id"] + [f"f_{n}" for n in range(300)]

    def __getitem__(self, investment_id: int) -> Investment:
        if investment_id not in self.investments:
            self.investments[investment_id] = Investment.empty(
                self.investment_df.loc[investment_id] if investment_id in self.investment_df.index else None,
                investment_id
            )
        return self.investments[investment_id]

    def extend(self, row: pd.Series):
        self[row["investment_id"]].features.extend(
            row[self.feature_cols], row["time_id"])
