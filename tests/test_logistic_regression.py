import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip('numpy')

from scripts.logistic_regression import LogisticRegression


def test_sigmoid():
    lr = LogisticRegression(np.zeros((1, 1)))
    assert pytest.approx(lr.sigmoid(0)) == 0.5


def test_probability_and_loss_and_update():
    lr = LogisticRegression(np.zeros((2, 1)))
    lr.a = np.array([1.0])
    lr.b = 0.0
    X = np.array([[0.0], [1.0]])
    preds = lr.probability(X)
    expected = lr.sigmoid(np.array([0.0, 1.0]))
    assert np.allclose(preds, expected)

    y = np.array([0, 1])
    loss = lr.loss(preds, y)
    expected_loss = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
    assert np.isclose(loss, expected_loss)

    lr.a_grad = np.array([0.1])
    lr.b_grad = 0.2
    a_prev, b_prev = lr.a.copy(), lr.b
    lr.update(0.5)
    assert np.allclose(lr.a, a_prev - 0.5 * 0.1)
    assert np.isclose(lr.b, b_prev - 0.5 * 0.2)
