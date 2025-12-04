import numpy as np

class JayaAlgorithm:
    def __init__(self, obj_func, dim, pop_size=20, iterations=100, bounds=(-10, 10)):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.iterations = iterations
        self.lb, self.ub = bounds

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def optimize(self):
        population = self.initialize_population()
        fitness = np.apply_along_axis(self.obj_func, 1, population)

        best_scores = []

        for t in range(self.iterations):
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)

            best = population[best_idx]
            worst = population[worst_idx]

            new_population = np.copy(population)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    new_val = population[i, j] \
                              + r1 * (best[j] - abs(population[i, j])) \
                              - r2 * (worst[j] - abs(population[i, j]))

                    # Apply bounds
                    new_population[i, j] = np.clip(new_val, self.lb, self.ub)

            new_fitness = np.apply_along_axis(self.obj_func, 1, new_population)

            # Accept improvement
            improved = new_fitness < fitness
            population[improved] = new_population[improved]
            fitness[improved] = new_fitness[improved]

            best_scores.append(np.min(fitness))

            print(f"Iteration {t+1}/{self.iterations}, Best Score: {best_scores[-1]}")

        return population[np.argmin(fitness)], best_scores
