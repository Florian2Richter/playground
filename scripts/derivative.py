import numpy as np
import matplotlib.pyplot as plt


def func(alpha, x):
    return alpha * np.sin(x)


def shift(array):
    # shift an array
    result_array = np.random.rand(len(array))
    result_array[0] = array[-1]
    result_array[1:] = array[:-1]
    return result_array


def derivative(image, dx):
    # f(x+dx)-f(x)/dx
    return (shift(image) - image) / dx


# calculate the derivative of some function
dx = 0.001
values = np.arange(0, 2 * 3.141 + dx, dx)


alpha = 1
image = func(alpha, values)
deriv = derivative(image, dx)
plt.plot(image)
plt.plot(deriv)
plt.show()
