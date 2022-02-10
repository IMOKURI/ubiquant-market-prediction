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


def test_extend_basic():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    assert s.length == 2
    assert s.values[0][0] == 1.0
    assert s.values[1][1] == 4.0


def test_extend_grow():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.ones((999, 2), dtype=np.float32)
    s.extend(array)

    assert s.length == 999

    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    assert s.length == 1001
    assert s.capacity == 1500


def test_last_n_basic():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    res = s.last_n(1)

    assert res.shape == (1, 2)
    assert res[0][1] == 4.0


def test_last_n_large():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    res = s.last_n(3)

    assert res.shape == (3, 2)
    assert res[0][0] == 0.0
    assert res[1][0] == 1.0
