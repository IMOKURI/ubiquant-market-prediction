import pandas as pd

from .investment import Investments


class Store:
    def __init__(self, investments: Investments):
        self.investments = investments

    @classmethod
    def empty(cls) -> 'Store':
        feature_cols = ["time_id", "investment_id"] + [f"f_{n}" for n in range(300)]
        empty_feature = pd.DataFrame(columns=feature_cols)

        investments = Investments(empty_feature)

        return cls(investments)
