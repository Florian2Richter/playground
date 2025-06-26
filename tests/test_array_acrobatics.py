import pytest

np = pytest.importorskip('numpy')

from scripts.array_acrobatics import convolution2D


def test_convolution2d_small():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    result = convolution2D(image, kernel)
    assert np.array_equal(result, np.array([[5]]))
