import numpy as np


def convolution2D(image, kernel):
    offset_x = kernel.shape[0] // 2
    offset_y = kernel.shape[1] // 2
    result = []
    for i in range(offset_x, image.shape[0] - offset_x):
        for j in range(offset_y, image.shape[1] - offset_y):
            # here convolve
            result.append(
                np.sum(
                    image[i - offset_x : i + offset_x, j - offset_y : j + offset_y]
                    * kernel
                )
            )
    return np.array(result)


example = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

print(convolution2D(example, kernel))
