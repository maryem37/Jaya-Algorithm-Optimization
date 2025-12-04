from jaya import JayaAlgorithm
import numpy as np

# Objective function: Sphere function
def sphere(x):
    return np.sum(x**2)

# Run Jaya on 1-dimensional optimization: minimize f(x) = x^2
optimizer = JayaAlgorithm(
    obj_func=sphere,
    dim=1,
    pop_size=20,
    iterations=100,
    bounds=(-10, 10)
)

best_solution, history = optimizer.optimize()

print("\nBest solution found:", best_solution)
print("Best fitness:", history[-1])
