import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the hyperbolic function
def hyperbola(x, y):
    return x**2 - y**2


# Define the tangential vector field on the surface
def tangential_vector_field(x, y):
    # Tangential vector field: F(x, y) = (-y, x)
    # This will ensure the vectors are tangential to the surface
    u = -y
    v = x
    return u, v


# Grid for x and y
x_val = np.linspace(-1, 1, 20)
y_val = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x_val, y_val)

# Compute z values on the hyperbolic surface
z = hyperbola(x, y)

# Compute the tangential vector field (u, v) at each point (x, y)
u, v = tangential_vector_field(x_val, y_val)
# Since the vectors are tangential, their z-component is 0
w = np.zeros_like(u)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
ax.plot_surface(x, y, z)

# Plot the tangential vector field
ax.quiver(x, y, z, u, v, w, length=0.2, color="red")

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.title("Tangential Vector Field on Hyperbolic Surface")
plt.savefig("hyperbolic_surface.png")
plt.show()
