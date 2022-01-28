import pandas as pd

from .investment import Investments


class Store:
    def __init__(self, investments: Investments):
        self.investments = investments
        self.last_time_id = None

    @classmethod
    def empty(cls) -> 'Store':
        feature_cols = ["time_id", "investment_id"] + [f"f_{n}" for n in range(300)]
        empty_feature = pd.DataFrame(columns=feature_cols)

        investments = Investments(empty_feature)

        return cls(investments)

    def append(self, row: pd.Series):
        if "time_id" not in row.index:
            # TODO: row_id のパースに失敗したときのハンドリング
            # last_time_id をうまく使えないだろうか
            row['time_id'] = row['row_id'].split('_')[0]

        # TODO: 最終的に catch_everything_in_kaggle をいれていく
        self.investments.extend(row)

        self.last_time_id = row["time_id"]
