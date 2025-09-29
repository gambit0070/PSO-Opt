#particle swarm optimization

import numpy as np

class PSO:
  def __init__(self, fitness, seed, dim, Xmin, Xmax, populationNumber, max_iter, timeScale, w = 1.0 , sigma_p = 2.0, sigma_g = 2.0):
    self.populationNumber = populationNumber
    self.dim = dim
    self.Fitness = fitness
    self.Xmin = Xmin
    self.Xmax = Xmax
    self.max_iter = max_iter
    self.rng = np.random.default_rng(seed)
    self.w = w
    self.sigma_p = sigma_p
    self.sigma_g = sigma_g
    self.timeScale = timeScale
    return 
  
  def Run(self):
    Xarray = np.zeros((self.populationNumber, self.dim))
    Varray = np.zeros((self.populationNumber, self.dim))
    XbestArray = np.zeros((self.populationNumber, self.dim))
    Gbest = []
    best_fitness = 0.0
    for i in range(self.populationNumber):
      for j in range(self.dim):
        Xarray[i][j] = (self.Xmax[j] - self.Xmin[j]) * self.rng.uniform(0,1) + self.Xmin[j]
        XbestArray[i][j] = Xarray[i][j]
      fit = self.Fitness(Xarray[i], self.timeScale)
      if len(Gbest) == 0 or self.Fitness(Gbest, self.timeScale) < fit:
        Gbest = Xarray[i].copy()
        best_fitness = fit
      for j in range(self.dim):
        d = self.Xmax[j] - self.Xmin[j]
        Varray[i][j] = self.rng.uniform(-d , d)

    for k in range(self.max_iter):
      for i in range(self.populationNumber):
        rp = self.rng.uniform(0, 1, size=(self.dim))
        rg = self.rng.uniform(0, 1, size=(self.dim))
        Varray[i] = self.w * Varray[i] + self.sigma_p * rp * (XbestArray[i] - Xarray[i])
        + self.sigma_g * rg * (Gbest - Xarray[i])

        Xarray[i] = Xarray[i] + Varray[i]

        Xarray[i] = np.clip(Xarray[i], self.Xmin, self.Xmax)


        fit = self.Fitness(Xarray[i], self.timeScale)
        if fit > self.Fitness(XbestArray[i], self.timeScale):
          XbestArray[i] = Xarray[i]
          if fit > self.Fitness(Gbest, self.timeScale):
            Gbest = Xarray[i].copy()
            best_fitness = fit
      print("fitness:", best_fitness)

    return Gbest, best_fitness
