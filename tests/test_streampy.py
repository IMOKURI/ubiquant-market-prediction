import numpy as np

from src.streampy import StreamPy


def test_construct():
    s = StreamPy.empty(["a", "b"], np.float32)

    assert s.values.shape == (1000, 2)
    assert s.values.dtype == np.float32
    assert s.length == 0
    assert s.default_value is None
    assert s.capacity == 1000
    assert s.columns == ["a", "b"]
