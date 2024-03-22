import numpy as np


def convolution2D(image, kernel):
    offset_x = kernel.shape[0] // 2
    offset_y = kernel.shape[1] // 2
    result = []
    for i in range(offset_x, image.shape[0] - offset_x):
        row = []
        for j in range(offset_y, image.shape[1] - offset_y):
            # here convolve
            image_snip = image[
                i - offset_x : i + offset_x + 1, j - offset_y : j + offset_y + 1
            ]
            row.append(np.sum(image_snip * kernel))
        result.append(row)

    return np.array(result)


example = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

result = convolution2D(example, kernel)
print(result)
