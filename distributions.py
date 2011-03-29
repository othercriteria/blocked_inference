# Experiments with EM estimation on HMM data.
# Daniel Klein, 3/29/2011

import numpy as np

class Laplace:
    def __init__(self, mu = 0.0, b = 1.0, max_b = 0.0):
        self.mu, self.b, self.max_b = mu, b, max_b
    
    def from_data(self, data, weights = None):
        if not weights is None:
            m = np.average(data, weights = weights)
            v = np.average((data - m)**2, weights = weights)
        else:
            m = np.average(data)
            v = np.average((data - m)**2)
        return Laplace(m, min(np.sqrt(v / 2), self.max_b), self.max_b)

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
        return (lambda x: c * np.exp(-abs(x - self.mu) / self.b))

class Normal:
    def __init__(self, mu = 0.0, sigma = 1.0, max_sigma = 0.0):
        self.m, self.s, self.max_s = mu, sigma, max_sigma
    
    def from_data(self, data, weights = None):
        if not weights is None:
            m = np.average(data, weights = weights)
            v = np.average((data - m)**2, weights = weights)
        else:
            m = np.average(data)
            v = np.average((data - m)**2)
        return Normal(m, min(np.sqrt(v), self.max_s), self.max_s)

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
        return (lambda x: c * np.exp(-0.5 * (x - self.m) ** 2 / (self.s ** 2)))

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
        def d(x):
            contrib = np.exp(-0.5 * (x - self.x) ** 2 / (h ** 2))
            return c * np.average(contrib, weights = self.w)
        return d
