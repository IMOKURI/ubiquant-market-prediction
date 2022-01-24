import os
import pandas as pd

from .investment import Investments


class Store:
    def __init__(self, investmens: Investments):
        self.investments = investmens

    @classmethod
    def empty(cls, data_dir: str) -> 'Store':
        investments_df = pd.read_csv(os.path.join(
            data_dir, "investments.csv"), index_col=1)  # type: pd.DataFrame

        investments = Investments(investments_df)

        return cls(investments)

    @classmethod
    def train(cls, data_dir: str, feature_dir: str = None, use_updated: bool = True) -> 'Store':
        instance = cls.empty(data_dir)

        stem = 'train_updated' if use_updated else 'train'
        if feature_dir is not None and os.path.exists(os.path.join(feature_dir, f'{stem}.f')):
            df = pd.read_feather(os.path.join(feature_dir, f'{stem}.f'))
        elif os.path.exists(os.path.join(data_dir, f'{stem}.f')):
            df = pd.read_feather(os.path.join(data_dir, f'{stem}.f'))
        else:
            df = pd.read_csv(os.path.join(data_dir, f'{stem}.csv'))

        return instance
