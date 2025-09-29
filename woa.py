import numpy as np

class WOA_Optimizer:
    ''' fitness: the fitness function as an input
        seed: seed of random generator
        dim: the number of hyperparameters or equivalently the dim of the hyperparameter vector.
        Xmin: the lower bounds for the hyperparameters, should has length == dim
        Xmax: the upper bounds for the hyperparameters, should has length == dim
        populationNumber: the number of population
        max_iter: iteration number 
    '''
    def __init__(self, fitness, seed, dim, Xmin, Xmax, populationNumber, max_iter, scale):
        self.X_array=[]
        self.max_iter = max_iter

        self.dim = dim 
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Fitness = fitness
        self.rng = np.random.default_rng(seed)
        self.n  = populationNumber #poplulation number
        self.timeScale = scale
        ##these are internal paramters of the algorithm
        self.a = 2 #a is linearly decreased from 2 to 0 over the course
        self.r = 0 #random vector in [0,1]
        self.A = 0 #2 * a * r - a
        self.C = 0 #2 * r
        self.l = 0 #random number in [-1,1]
        self.p = 0 #random number in [0,1]
        self.b = 1 #const to define log spiral

        return
    
    #in each iteration, update the internal parameters
    def Update(self):
        self.a =  self.a -  2.0 /self.max_iter
        self.r = self.rng.uniform(0, 1)
        self.A = self.a * self.r * 2 - self.a
        self.C = self.r * 2.0
        self.l = self.rng.uniform(-1, 1)
        self.p = self.rng.uniform(0, 1)

        return

    #corresponding to the first equation in the algo
    def Equation1(self, X , X_target):
        D = np.abs(X_target * self.C - X)
        return D

    #corresponding to the second equation in the algo
    def Equation2(self,X, X_target):
        D_ = np.abs(X_target - X)
        X_output = D_ * np.exp(self.b * self.l) * np.cos(2* np.pi * self.l) + X_target
        return X_output

    #corresponding to the third equation in the algo
    def Equation3(self, X,X_rand):
        D = np.abs(X_rand * self.C - X)
        X_output = X_rand - D * self.A
        return X_output

    #in case of out of bound
    def Amend(self, X_array):
        for X in X_array:
            for j in range(self.dim):
                if X[j] < self.Xmin[j]:
                    X[j] = self.Xmin[j]
                elif X[j] > self.Xmax[j]:
                    X[j] = self.Xmax[j]
        return

    #generate a random candidate
    def RandomAgent(self):
        Xs = np.zeros(self.dim)
        for i in range(0, self.dim):
            Xs[i] = (self.Xmax[i] - self.Xmin[i]) * self.rng.uniform(0,1) + self.Xmin[i]
        return Xs

    #Initialize the whale population Xi(i = 1, 2, ..., n)
    def RandomPopulation(self):
        X_array = np.zeros((self.n, self.dim))
        for i in range(self.n):
            X_array[i] = self.RandomAgent()
        return X_array

    #Calculate the fitness of each search agent and find the best one
    def FindBestAgent(self, X_array ,X_target):
        fitness = 0
        if X_target is not None:
            fitness_target = self.Fitness(X_target, self.timeScale)
        else:
            fitness_target = self.Fitness(X_array[0], self.timeScale)
            X_target = X_array[0]
        for X in X_array:
            fitness = self.Fitness(X, self.timeScale)
            if fitness > fitness_target:
                #Update X* if there is a better solution
                fitness_target = fitness
                X_target = X
        X_target = X_target.copy()
        return X_target, fitness_target


    #main function, run the algorithm
    def Run(self):
        X_target = None
        #Initialize the whales population Xi(i = 1, 2, ..., n) 
        X_array = self.RandomPopulation()
        #Calculate the fitness of each search agent
        #X*=the best search agent
        X_target, fit = self.FindBestAgent(X_array, None)

        fits = []
        t = 0
        while t < self.max_iter:
            for i in range(len(X_array)):
                X = X_array[i]
                X1 = None
                self.Update()
                if self.p < 0.5:
                    if abs(self.A) < 1:
                        X1 = self.Equation1(X, X_target)
                    else:
                        X_rand = self.RandomAgent()
                        X1 = self.Equation3(X,X_rand)
                else:
                  X1 = self.Equation2(X,X_target)

                X_array[i] = X1

            #Check if any search agent goes beyond the search space and amend its position
            self.Amend(X_array)

            #Calculate the fitness of each search agent
            #Update X* if there is a better solution
            X_target, fit = self.FindBestAgent(X_array, X_target)
            # print(f"{t} current_best_fitness:", fit)
            fits.append(fit)

            t = t + 1

        return  X_target, fit, fits
