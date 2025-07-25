# load data for logistic regression
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# now start with the logistic regression ;)


class LogisticRegression:
    def __init__(self, X):
        self.a = np.random.rand(X.shape[1]) - 0.5
        self.b = np.random.rand()
        self.a_grad = 0
        self.b_grad = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def probability(self, X):
        z = np.dot(X, self.a) + self.b
        return self.sigmoid(z)

    def loss(self, pred, y):
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def gradient(self, pred, X, y):
        N, D = X.shape
        a_grad = np.dot(X.transpose(), (pred - y)) / N
        b_grad = np.mean(pred - y)

        return a_grad, b_grad

    def update(self, stepsize):
        self.a = self.a - stepsize * self.a_grad
        self.b = self.b - stepsize * self.b_grad

    def train(self, X, y, stepsize=0.01):
        pred = self.probability(X)
        self.a_grad, self.b_grad = self.gradient(pred, X, y)
        self.update(stepsize)

    def accuracy(self, X, y, threshold=0.5):
        probs = self.probability(X)
        predictions = np.array(probs > threshold, dtype=int)
        corrects = np.sum(y == predictions)
        return corrects / len(y)


def load_data():
    """Load and return the breast cancer dataset used in the examples."""
    csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breast_cancer.csv")
    df = pd.read_csv(csv_file_path)
    df["Class"] = df["Class"].replace({2: 0, 4: 1})
    X = df.drop(columns="Class").to_numpy()
    y = df["Class"].to_numpy()
    return X, y


def main():
    X, y = load_data()

    indices = list(range(len(y)))
    random.shuffle(indices)
    split_idx = int(len(y) * 0.6)
    X_train, y_train = X[indices[:split_idx]], y[indices[:split_idx]]
    X_test, y_test = X[indices[split_idx:]], y[indices[split_idx:]]

    logistic_regressor = LogisticRegression(X)
    pred = logistic_regressor.probability(X)
    print(f"loss is {logistic_regressor.loss(pred,y)}")

    train_losses = []
    test_accuracies = []
    for i in range(100000):
        logistic_regressor.train(X_train, y_train)
        if i % 1000 == 0:
            train_loss = logistic_regressor.loss(
                logistic_regressor.probability(X_train), y_train
            )
            test_accuracy = logistic_regressor.accuracy(X_test, y_test)
            print(f"Training step {i}")
            print(f"training loss is {train_loss}")
            print(f"test accuracy is {test_accuracy}")
            train_losses.append(train_loss)
            test_accuracies.append(test_accuracy)
    plt.plot(train_losses)
    plt.plot(test_accuracies)
    plt.show()


if __name__ == "__main__":
    main()
