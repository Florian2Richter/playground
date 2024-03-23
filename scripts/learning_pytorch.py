import torch
import math
from matplotlib import pyplot as plt

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

# plt.plot(y)
# plt.show()


# define trainable parameters
a = torch.randn((),dtype=dtype, requires_grad=True)
b = torch.randn((),dtype=dtype, requires_grad=True)
c = torch.randn((),dtype=dtype, requires_grad=True)
d = torch.randn((),dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(10000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.

    loss = torch.sum((y-y_pred)**2)
    # print(f"loss is {loss.item()}")

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()
    # print(a.grad)

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate*a.grad
        b -= learning_rate*b.grad
        c -= learning_rate*c.grad
        d -= learning_rate*d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

plt.plot(y_pred.cpu().detach().numpy())
plt.plot(y.cpu().detach().numpy())
plt.show()

    
