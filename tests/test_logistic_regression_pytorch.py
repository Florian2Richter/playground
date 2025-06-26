import importlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip('torch')

lr_mod = importlib.import_module('scripts.logistic_regression_pytorch')


def test_sigmoid():
    model = lr_mod.LogisticRegression(torch.zeros((1, 1)))
    assert torch.isclose(model.sigmoid(torch.tensor(0.0)), torch.tensor(0.5))


def test_probability_loss_update():
    model = lr_mod.LogisticRegression(torch.zeros((2, 1)))
    model.a = torch.tensor([1.0], dtype=lr_mod.dtype, device=lr_mod.device, requires_grad=True)
    model.b = torch.tensor(0.0, dtype=lr_mod.dtype, device=lr_mod.device, requires_grad=True)
    X = torch.tensor([[0.0], [1.0]], dtype=lr_mod.dtype, device=lr_mod.device)
    preds = model.probability(X)
    expected = model.sigmoid(torch.tensor([0.0, 1.0], dtype=lr_mod.dtype, device=lr_mod.device))
    assert torch.allclose(preds, expected)

    y = torch.tensor([0.0, 1.0], dtype=lr_mod.dtype, device=lr_mod.device)
    loss_val = model.loss(preds, y)
    expected_loss = -torch.mean(y * torch.log(preds) + (1 - y) * torch.log(1 - preds))
    assert torch.allclose(loss_val, expected_loss)

    model.a.grad = torch.tensor([0.1], dtype=lr_mod.dtype, device=lr_mod.device)
    model.b.grad = torch.tensor(0.2, dtype=lr_mod.dtype, device=lr_mod.device)
    a_before = model.a.clone()
    b_before = model.b.clone()
    model.update(0.5)
    assert torch.allclose(model.a, a_before - 0.5 * 0.1)
    assert torch.allclose(model.b, b_before - 0.5 * 0.2)
