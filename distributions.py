# Experiments with EM estimation on HMM data.
# Daniel Klein, 3/29/2011

import numpy as np

# Core methods are: __init__, from_data, density

class Product:
    def __init__(self, dists):
        self.dists = dists

    def from_data(self, data, weights = None):
        return Product([dist.from_data(data[:,i], weights)
                        for i, dist in enumerate(self.dists)])

    def mean(self):
        return [dist.mean() for dist in self.dists]

    def sd(self):
        return [dist.sd() for dist in self.dists]

    def density(self):
        densities = [dist.density() for dist in self.dists]
        def d(xs):
            def d_single(x):
                return np.product([density(x_ind)
                                   for density, x_ind in zip(densities, x)])
            return map(d_single, xs)
        return d

class Bernoulli:
    def __init__(self, params = {'p': 0.5}):
        self.p = params['p']

    def from_data(self, data, weights = None):
        p = np.average(data, weights = weights)
        return Bernoulli({'p': p})

    def mean(self):
        return self.p

    def sd(self):
        return np.sqrt(self.p * (1.0 - self.p))

    def display(self):
        return 'Berboulli(p = %.2f)' % self.p

    def sample(self):
        return ((np.random.random() < self.p) and 1.0 or 0.0)

    def density(self):
        def d(xs):
            return (self.p * xs + (1.0 - self.p) * (1.0 - xs))
        return d

class Laplace:
    def __init__(self, params = {'mu': 0.0, 'b': 1.0}, max_b = 0.0):
        self.mu, self.b, self.max_b = params['mu'], params['b'], max_b
    
    def from_data(self, data, weights = None):
        m = np.average(data, weights = weights)
        v = np.average((data - m)**2, weights = weights)
        b = min(np.sqrt(v / 2), self.max_b)
        return Laplace({'mu': m, 'b': b}, self.max_b)

    def mean(self):
        return self.mu

    def sd(self):
        return np.sqrt(2) * self.b

    def display(self):
        return 'Laplace(mu = %.2f, b = %.2f)' % (self.mu, self.b)

    def sample(self):
        return np.random.laplace(self.mu, self.b)

    def density(self):
        c = 1 / (2 * self.b)
        def d(xs):
            return c * np.exp(-abs(xs - self.mu) / self.b)
        return d

class Normal:
    def __init__(self, params = {'m': 0.0, 's': 1.0}, max_sigma = 0.0):
        self.m, self.s, self.max_s = params['m'], params['s'], max_sigma
    
    def from_data(self, data, weights = None):
        m = np.average(data, weights = weights)
        v = np.average((data - m)**2, weights = weights)
        s = min(np.sqrt(v), self.max_s)
        return Normal({'m': m, 's': s}, self.max_s)

    def mean(self):
        return self.m

    def sd(self):
        return self.s

    def display(self):
        return 'Normal(mu = %.2f, sigma = %.2f)' % (self.m, self.s)

    def sample(self):
        return np.random.normal(self.m, self.s)

    def density(self):
        c = (1 / np.sqrt(2 * np.pi * self.s))
        def d(xs):
            return c * np.exp(-0.5 * (xs - self.m) ** 2 / (self.s ** 2))
        return d

class Kernel:
    def __init__(self, x = None, w = None, h = 1.0):
        self.x, self.w, self.h = x, w, h

    def __copy__(self):
        return Kernel(self.h)

    def from_data(self, data, weights = None):
        return Kernel(data, weights, self.h)

    def mean(self):
        return np.average(self.x, weights = self.w)

    def sd(self):
        v = np.average((self.x - self.mean()) ** 2, weights = self.w)
        return np.sqrt(v + self.h_adapt() ** 2)

    def h_adapt(self):
        return self.h * max(1.0, sum(self.w)) ** (-0.2)

    def display(self):
        return 'Kernel density (n ~ %.2f, mean = %.2f, sd = %.2f; h = %.2f)' \
               % (sum(self.w), self.mean(), self.sd(), self.h_adapt())

    def density(self):
        h = self.h_adapt()
        c = (1 / np.sqrt(2 * np.pi * h))
        def d(xs):
            def d_single(x):
                contrib = np.exp(-0.5 * (x - self.x) ** 2 / (h ** 2))
                return np.average(contrib, weights = self.w)
            return c * (np.vectorize(d_single))(xs)
        return d
