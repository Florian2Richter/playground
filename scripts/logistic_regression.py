# load data for logistic regression
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

csv_file_path = os.path.dirname(os.path.abspath(__file__)) + "/breast_cancer.csv"
df = pd.read_csv(csv_file_path)

# try to replace the values (2=benign) and (4=malignant) by 0 and 1
df["Class"] = df["Class"].replace({2: 0, 4: 1})

# now start with the logistic regression ;)


class LogisticRegression:
    def __init__(self, a_0, b_0):
        self.a = a_0
        self.b = b_0
        self.a_grad = 0
        self.b_grad = 0

    def logit(self, x):
        return 1 / (1 + np.exp(-(self.a * x + self.b)))

    def loss(self, pred, y):
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def gradient(self, pred, y, x):

        a_grad = np.mean((pred - y) * x)
        b_grad = np.mean(pred - y)

        return a_grad, b_grad

    def update(self, stepsize):
        self.a = self.a * stepsize * self.a_grad
        self.b = self.b * stepsize * self.b_grad
        return

    def train(self, x, y, stepsize=0.001):
        pred = self.logit(x)
        print("loss is")
        print(self.loss(pred, y))
        self.a_grad, self.b_grad = self.gradient(pred, x, y)
        self.update(stepsize)
        return


# initialize
logitistic_regressor = LogisticRegression(np.random.rand(), np.random.rand())
x = df["Mitoses"]
y = df["Class"]
pred = logitistic_regressor.logit(x)
# testplot
print(logitistic_regressor.loss(pred, y))

# now train the regression

for i in range(10):
    print(f"Training step {i}")
    logitistic_regressor.train(x, y)

# y = np.array([1, 0, 1, 1, 0])  # Ground truth labels
# pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Model predictions
#
# plt.show()
