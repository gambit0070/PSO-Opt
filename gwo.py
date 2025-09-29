import numpy as np

class GreyWolfOptimiser:
    def __init__(self, fitness_function, max_iter, num_wolves, dim, minx, maxx, seed=0, fitness_args=None):
        self.fitness = fitness_function
        self.fitness_args = fitness_args if fitness_args is not None else ()
        self.max_iter = max_iter
        self.num_wolves = num_wolves
        self.dim = dim # dimension
        self.minx = np.array(minx)
        self.maxx = np.array(maxx)
        self.rng = np.random.default_rng(seed)

        # Initial wolf positions
        self.positions = self.rng.uniform(minx, maxx, size=(num_wolves, dim))
        self.fitness_values = np.array([fitness_function(pos, *self.fitness_args) for pos in self.positions])
        
        # Initialise alpha, beta, and delta positions
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_score = 0
        self.beta_score = 0
        self.delta_score = 0

    def update_leader_positions(self):
        # Sort wolves from maximum fitness to minimum fitness
        sorted_indices = np.argsort(self.fitness_values)[::-1]
        
        # Update alpha, beta, and delta positions
        self.alpha_wolf = self.positions[sorted_indices[0]] # best wolf
        self.beta_wolf = self.positions[sorted_indices[1]] # second best wolf
        self.delta_wolf = self.positions[sorted_indices[2]] # third best wolf
        
        # Update their scores
        self.alpha_score = self.fitness_values[sorted_indices[0]] # best fitness value
        self.beta_score = self.fitness_values[sorted_indices[1]] # second best fitness value
        self.delta_score = self.fitness_values[sorted_indices[2]] # third best fitness value

    def update_wolf_positions(self, iteration):
        # a linearly decreases from 2 to 0
        a = 2 - iteration * (2 / self.max_iter)
        
        for i in range(self.num_wolves):
            # Update each wolf's position
            for j in range(self.dim):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                # D_alpha = distance between alpha wolf and current wolf
                D_alpha = abs(C1 * self.alpha_wolf[j] - self.positions[i, j])
                X1 = self.alpha_wolf[j] - A1 * D_alpha
                
                r1 = self.rng.random()
                r2 = self.rng.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                # D_beta = distance between beta wolf and current wolf
                D_beta = abs(C2 * self.beta_wolf[j] - self.positions[i, j])
                X2 = self.beta_wolf[j] - A2 * D_beta
                
                r1 = self.rng.random()
                r2 = self.rng.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                # D_delta = distance between delta wolf and current wolf
                D_delta = abs(C3 * self.delta_wolf[j] - self.positions[i, j])
                X3 = self.delta_wolf[j] - A3 * D_delta
                
                # Update position
                self.positions[i, j] = (X1 + X2 + X3) / 3
                
                # Keep position within bounds
                self.positions[i, j] = np.clip(self.positions[i, j], self.minx[j], self.maxx[j])
            
            # Update the fitness value
            self.fitness_values[i] = self.fitness(self.positions[i], *self.fitness_args)

    def run(self):
        # Initialise leader (alpha, beta, delta) positions
        self.update_leader_positions()
        
        for iteration in range(self.max_iter):
            # Update wolf positions
            self.update_wolf_positions(iteration)
            
            # Update leader positions
            self.update_leader_positions()
        
        return self.alpha_wolf, self.alpha_score

# TESTING
# def fitness_function(x):
#     return sum(x**2)  # sum of squares

# gwo = GreyWolfOptimiser(
#     fitness_function=fitness_function,
#     max_iter=100,
#     num_wolves=30,
#     dim=5,
#     minx=-10,
#     maxx=10
# )   

# best_solution, best_fitness = gwo.run()
# print(f"Best solution: {best_solution}")
# print(f"Best fitness: {best_fitness}")