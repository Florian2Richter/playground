import numpy as np


def convolution_1d(input, kernel):
    d = input.shape[0]  # dimension of input
    k = kernel.shape[0]  # dimension of kernel
    result = []
    for index in range(0, d - k + 1):
        result.append(input[index : index + k] @ kernel)
    return np.array(result)


# here comes the convolution
def test_convolution_1d():
    input = np.array([1, 0, 1, 0])
    kernel = np.array([1, 1])
    result = np.array([1, 1, 1])
    prediction = convolution_1d(input, kernel)
    assert np.array_equal(result, prediction), (
        "result of the calculation is " + str(prediction) + " instead of " + str(result)
    print('test passed')
    )
test_convolution_1d()
