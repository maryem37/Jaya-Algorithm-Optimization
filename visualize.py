import matplotlib.pyplot as plt
from jaya import JayaAlgorithm
import numpy as np

def sphere(x):
    return np.sum(x**2)

optimizer = JayaAlgorithm(sphere, dim=1)
best, history = optimizer.optimize()

plt.plot(history)
plt.title("Convergence of Jaya Algorithm")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
