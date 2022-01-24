from typing import Dict, Optional

import pandas as pd


class Investment:
    def __init__(self, row: Optional[pd.Series], investment_id: int, time_id: int = None):
        self.row = row
        self.investment_id = investment_id
        self.time_id = time_id

    @classmethod
    def empty(cls, row: Optional[pd.Series], investment_id: int) -> 'Investment':
        return Investment(row, investment_id)


class Investments:
    def __init__(self, investment_df: pd.DataFrame):
        self.investment_df = investment_df
        self.investments = {}  # type: Dict[int, Investment]

    def __getitem__(self, investment_id: int) -> Investment:
        if investment_id not in self.investments:
            self.investments[investment_id] = Investment.empty(
                self.investment_df.loc[investment_id] if investment_id in self.investment_df.index else None, investment_id)

        return self.investments[investment_id]
