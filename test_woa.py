from woa import *

# sphere function
def fitness_sphere(X):
    fitness_value = 0.0
    for i in range(len(X)):
        xi = X[i]
        fitness_value -= (xi * xi)
    return fitness_value


if __name__ == "__main__":
    dim = 4
    Xmin = [0] * dim
    Xmax = [10] * dim
    woa = WOA_Optimizer(fitness_sphere, 42, 4, Xmin, Xmax, 200, 500)

    X_target = woa.Run()
    fit = woa.Fitness(X_target)
    print("best_result:", X_target)
    print("fitness:" , fit)