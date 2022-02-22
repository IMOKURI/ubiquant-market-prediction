# https://github.com/nyanp/streamdf/blob/main/streamdf/streamdf.py

from typing import Any, List, Type

import numpy as np
import pandas as pd
from nptyping import NDArray


class StreamPy:
    """
    Numpy array that aim to improve performance to extend.
    Holds the number of rows currently stored and expands it before it fills up.
    """

    def __init__(
        self,
        values: NDArray[(Any, Any), Any],  # 2-dim with any size and any dtype
        columns: List[str],
        length: int = 0,
        default_value: Any = None,
    ):
        """
        Args:
            values (np.ndarray): Numpy array to store data.
            length (int): Number of data currently stored.
            default_value (int): Value to be inserted when an exception is raised.
        """
        self.values = values
        self.dtype = values.dtype
        self.length = length
        self.default_value = default_value

        self.capacity = values.shape[0]
        self.columns = columns

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return self.length

    @property
    def shape(self):
        return self.length, self.values.shape

    @property
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.columns)

    @classmethod
    def empty(
        cls,
        columns: List[str],
        dtype: Type,
        default_value: Any = None,
        capacity: int = 1000,
    ):
        values = np.zeros((capacity, len(columns)), dtype=dtype)

        return cls(values, columns, default_value=default_value)

    def extend(self, array: NDArray[(Any, Any), Any]):
        n_row, n_col = array.shape

        assert n_col == len(
            self.columns
        ), "The number of columns in the array does not match the number of column names."

        if self.length + n_row >= self.capacity:
            self._grow(self.length + n_row)

        # TODO: Exception 発生したときは default_value か
        self.values[self.length : self.length + n_row] = array
        self.length += n_row

    def last_n(self, n: int) -> NDArray[(Any, Any), Any]:
        res = self.values[max(self.length - n, 0) : self.length]

        if len(res) == n:
            return res

        res = np.concatenate([np.zeros((n - len(res), len(self.columns)), dtype=res.dtype), res])
        return res

    def _grow(self, min_capacity):
        capacity = max(int(1.5 * self.capacity), min_capacity)
        new_data_len = capacity - self.capacity
        assert new_data_len > 0

        self.values = np.concatenate(
            [self.values, np.empty((new_data_len, len(self.columns)), dtype=self.values.dtype)]
        )
        self.capacity += new_data_len
