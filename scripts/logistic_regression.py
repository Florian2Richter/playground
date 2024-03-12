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


def logit(x, c):
    return c + 1 / (1 + np.exp(-x))


def binary_cross_entropy(y, pred):
    y * np.log(pred) + (1 - y) * np.log(1 - pred)


y = np.array([1, 0, 1, 1, 0])  # Ground truth labels
pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Model predictions

result = y * pred

print(result)
