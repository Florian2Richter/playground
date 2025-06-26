import importlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip('numpy')
plt = pytest.importorskip('matplotlib.pyplot')


def test_shift_and_derivative(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    module = importlib.import_module('scripts.derivative')
    arr = np.array([1, 2, 3])
    shifted = module.shift(arr)
    assert np.array_equal(shifted, np.array([3, 1, 2]))
    deriv = module.derivative(arr, 1)
    expected = shifted - arr
    assert np.array_equal(deriv, expected)
