import random
import numpy as np
from GPyOpt.util.general import best_value

import rospy

class Sampling():
    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        self.count = 0

    def init(self, X0=None):
        raise Exception('Implementation Error')

    def step(self):
        raise Exception('Implementation Error')

    def update(self, Y_new):
        raise Exception('Implementation Error')

    def _compute_results(self):
        raise Exception('Implementation Error')

class GeneticAlgorithmSamling():
    def __init__(self, bounds, n_params=3, n_pops=5, toursize=2):
        self.bounds = bounds
        self.n_params = n_params
        self.n_pops = n_pops
        self.toursize = toursize
        self.X = None 
        self.Y = None 
        self.count = 1
        self.idx = 0
        self.pop = np.zeros((n_pops, n_params))

    def init(self, X0=None):
        init_pop = 1
        X = self.n_pops*np.random.rand(init_pop, self.n_params)
        self.X = np.array(X)
        return X[0,:]

    def step(self):
        X = self.pop[self.idx, :]
        self.X = np.vstack((self.X, X))
        self.idx += 1
        return X

    def update(self, Y_new):
        if self.Y is None:
            self.Y = np.array([Y_new])
        else:
            self.Y = np.hstack((self.Y, Y_new))
        self._compute_results()

        if self.count%self.n_pops == 0:
            self._update()
        
        self.count += 1

    def _update(self):
        start_idx = self.count-self.n_pops+1
        pop = self.X[start_idx:, :]
        fitnesses = self.Y[start_idx:]

        idx = np.argmax(fitnesses)
        best_chromosome = pop[idx, :] 

        players = np.ceil(self.n_pops*np.random.rand(self.n_pops, self.toursize)).astype(int)-1
        scores = np.array([[fitnesses[idx] for idx in player] for player in players])
        index = np.argmax(scores.transpose(), axis=0)
        pind = np.zeros((1,self.n_pops))

        parent = np.zeros((self.n_pops, self.n_params))
        for i in range(self.n_pops):
            pind[0,i] = players[i, index[i]]
            parent[i, :] = pop[int(pind[0,i]), :]

        child = self._cross(parent)

        self.pop = self._mutate(child)

        self.pop[0, :] = best_chromosome
        self.idx = 0

    def _compute_results(self):
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]

    def _cross(self, parent, cross_prob=0.8):
        n_pops, n_params = parent.shape
        offspring = np.zeros((n_pops, n_params))

        for i in range(int(n_pops/2)):
            if cross_prob > np.random.rand():
                r = np.random.randint(n_params)
                offspring[i, :r] = parent[i, :r]
                offspring[i, r:] = parent[i+1, r:]
                offspring[i+1, :r] = parent[i+1, :r]
                offspring[i+1, r:] = parent[i, r:]
            else:
                offspring[i, :] = parent[i, :]
                offspring[i+1, :] = parent[i+1, :]

        return offspring

    def _mutate(self, offspring, mutate_prob=0.35, param_min=10, param_range=20):
        n_pops, n_params = offspring.shape
        mutated_offs = offspring.copy()

        mut = np.round(mutate_prob*n_pops*n_params).astype(int)

        for i in range(mut):
            x = np.ceil(n_pops*np.random.rand()).astype(int)-1
            y = np.ceil(n_params*np.random.rand()).astype(int)-1
            mutated_offs[x,y] = param_min + param_range*np.random.rand()

        return mutated_offs

class RandomSampling():
    def __init__(self, bounds, X=None, Y=None):
        self.bounds = bounds 
        self.X = X
        self.Y = Y

    def init(self, X0=None):
        x1 = random.uniform(self.bounds[0]['domain'][0], self.bounds[0]['domain'][1])
        x2 = random.uniform(self.bounds[1]['domain'][0], self.bounds[1]['domain'][1])
        x3 = random.uniform(self.bounds[2]['domain'][0], self.bounds[2]['domain'][1])
        X = np.array([x1, x2, x3])
        self.X = np.array([X])
        return X

    def step(self):
        x1 = random.uniform(self.bounds[0]['domain'][0], self.bounds[0]['domain'][1])
        x2 = random.uniform(self.bounds[1]['domain'][0], self.bounds[1]['domain'][1])
        x3 = random.uniform(self.bounds[2]['domain'][0], self.bounds[2]['domain'][1])
        X = np.array([x1, x2, x3])
        self.X = np.vstack((self.X, X))
        return X

    def update(self, Y_new):
        if self.Y is None:
            self.Y = np.array([Y_new])
        else:
            self.Y = np.hstack((self.Y, Y_new))
        self._compute_results()

    def _compute_results(self):
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]

class BanditReinforcementLearning():
    def __init__(self, bounds, X=None, Y=None, epsilon=0.01, meshsize=10):
        self.bounds = bounds
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.meshsize = meshsize
        self.qEst = np.zeros((meshsize, meshsize, meshsize)) # Replace this w/ NN
        self.num = np.zeros((meshsize,meshsize,meshsize))
        Xs = np.linspace(self.bounds[0]['domain'][0], self.bounds[0]['domain'][1], meshsize)
        Ys = np.linspace(self.bounds[1]['domain'][0], self.bounds[1]['domain'][1], meshsize)
        Zs = np.linspace(self.bounds[2]['domain'][0], self.bounds[2]['domain'][1], meshsize)
        self.Xs, self.Ys, self.Zs = np.meshgrid(Xs, Ys, Zs)
        self.idx3d = (0,0,0)
        self.idx = 0

    def init(self):
        X = self._random_X()
        self.X = np.array([X])
        return X

    def _random_X(self):
        x1 = random.uniform(self.bounds[0]['domain'][0], self.bounds[0]['domain'][1])
        x2 = random.uniform(self.bounds[1]['domain'][0], self.bounds[1]['domain'][1])
        x3 = random.uniform(self.bounds[2]['domain'][0], self.bounds[2]['domain'][1])
        return np.array([x1, x2, x3])

    def step(self):
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                X = self._random_X()
                self.X = np.vstack((self.X, X))
                return X

        self.idx3d = np.unravel_index(np.argmax(self.qEst), self.qEst.shape)
        self.idx = np.argmax(self.qEst)
        X = np.array([
            self.Xs.flat[self.idx],
            self.Ys.flat[self.idx],
            self.Zs.flat[self.idx]
        ])
        self.X = np.vstack((self.X, X))
        
        return X

    def update(self, Y_new):
        if self.Y is None:
            self.Y = np.array([Y_new])
        else:
            self.Y = np.hstack((self.Y, Y_new))
        self.num[self.idx3d] += 1
        self.qEst[self.idx3d] += (Y_new-self.qEst[self.idx3d]) / self.num[self.idx3d]

    def _compute_results(self):
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]

class NewtonsMethod():
    def __init__(self, f, jacob, bounds, alpha=0.01):
        super().__init__(f)
        self.alpha = alpha
        self.jacob = jacob
        self.bounds = bounds
        self.dim = len(self.bounds)

    def _init_design_chooser(self, init_points_count=1):
        self.X = np.zeros((init_points_count, self.dim))
        self.Y = np.zeros((init_points_count, 1))

        for dim in range(0, self.dim):
            self.X[:, dim] = np.random.uniform(
                self.bounds[dim][0], self.bounds[dim][1], size=init_points_count).reshape((-1, 1)
                )

        for idx in range(init_points_count):
            self.Y[idx] = self.evaluate_objective(self.X[idx,:])

        self.X_new = self.X[-1]
        self.Y_new = self.Y[-1]

    def _compute_next_evaluations(self):
        return self.X_new - self.alpha * self.jacob(self.X_new)

    def evaluate_objective(self, X):
        return self.f(X)

    def run_optimization(self, max_iter=100, verbosity=False):
        self._init_design_chooser(init_points_count=1)

        while(max_iter > self.count):
            if np.abs(np.sum(self.jacob(self.X_new))) < 0.01:
                break
            self.X_new = self._compute_next_evaluations()
            self.Y_new = self.evaluate_objective(self.X_new)

            self.X = np.vstack((self.X, self.X_new))
            self.Y = np.vstack((self.Y, self.Y_new))

            self.count += 1

        return self.count