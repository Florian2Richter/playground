import numpy as np


def convolution_1d(input_array, kernel):
    """
    Perform a 1D convolution on a given input array with a specified kernel.

    Args:
    input_array (np.ndarray): The input 1D array.
    kernel (np.ndarray): The convolution kernel (1D array).

    Returns:
    np.ndarray: The result of the convolution operation.
    """
    input_length = input_array.shape[0]  # Dimension of the input array
    kernel_length = kernel.shape[0]  # Dimension of the kernel

    # Initialize the result list
    result = []

    # Perform the convolution operation
    for index in range(input_length - kernel_length + 1):
        convolved_value = np.dot(input_array[index : index + kernel_length], kernel)
        result.append(convolved_value)

    return np.array(result)


def test_convolution_1d():
    """
    Test the convolution_1d function with a specific input and kernel.
    Asserts that the result of convolution_1d matches the expected output.
    """
    input_array = np.array([1, 0, 1, 0])
    kernel = np.array([1, 1])
    expected_result = np.array([1, 1, 1])

    # Perform convolution
    prediction = convolution_1d(input_array, kernel)

    # Assert and check the result
    assert np.array_equal(
        expected_result, prediction
    ), f"Result of the calculation is {prediction} instead of {expected_result}"

    print("Test passed.")


# Call the test function
test_convolution_1d()
