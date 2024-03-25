import pandas as pd
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import torch

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

csv_file_path = os.path.dirname(os.path.abspath(__file__)) + "/breast_cancer.csv"
df = pd.read_csv(csv_file_path)

# try to replace the values (2=benign) and (4=malignant) by 0 and 1
df["Class"] = df["Class"].replace({2: 0, 4: 1})

# initialize data 
X_np = df.drop(columns="Class").to_numpy()  # get the whole dat

# now normalize the data-> tomorrow
y_np = df["Class"].to_numpy()

X = torch.from_numpy(X_np).to(device).float()
y = torch.from_numpy(y_np).to(device).float()

class LogisticRegression:
    def __init__(self, X):
        self.a = torch.rand(X.shape[1], dtype=dtype) - 0.5  # Step 1 & 3
        self.a = self.a.to(device)  # Step 2
        self.a.requires_grad_(True)  # Step 4

        self.b = torch.rand((),dtype = dtype, requires_grad=True , device=device)

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def probability(self, X):
        z = torch.matmul(X, self.a) + self.b
        return self.sigmoid(z)

    def loss(self, pred, y):
        return -torch.mean(y * torch.log(pred) + (1 - y) * torch.log(1 - pred))

    def update(self, stepsize=1e-3):
        with torch.no_grad():
            self.a -= stepsize * self.a.grad
            self.b -= stepsize * self.b.grad
            self.a.grad = None
            self.b.grad = None

    def train(self, X, y):
        self.loss(self.probability(X),y).backward()

def accuracy(predictor, X, y, threshold=0.5):
    probs = predictor.probability(X).to_numpy()
    predictions = np.array(probs > threshold, dtype=int)
    corrects = np.sum(y == predictions)
    return corrects / len(y)

predictor = LogisticRegression(X)
for step in range(500):
    predictor.train(X,y)
    predictor.update()
    loss = predictor.loss(predictor.probability(X),y)
    print(f"loss is {loss}")


