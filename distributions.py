# Experiments with EM estimation on HMM data.
# Daniel Klein, 3/29/2011

import numpy as np
import numpy.linalg as la
from scipy.stats import norm

# Core methods are: __init__, from_data, density

class Product:
    def __init__(self, dists):
        self.dists = dists

    def from_data(self, data, weights = None):
        for i, dist in enumerate(self.dists):
            dist.from_data(data[:,i], weights)

    def mean(self):
        return [dist.mean() for dist in self.dists]

    def sd(self):
        return [dist.sd() for dist in self.dists]

    def density(self, xs):
        num_in, num_prod = xs.shape
        dens_mat = np.empty((num_in, num_prod))
        for j, dist in enumerate(self.dists):
            dens_mat[:,j] = dist.density(xs[:,j])
        return np.product(dens_mat, axis = 1)

class Bernoulli:
    def __init__(self, p = 0.5):
        self.p = p

    def from_data(self, data, weights = None):
        self.p = np.average(data, weights = weights)

    def mean(self):
        return self.p

    def sd(self):
        return np.sqrt(self.p * (1.0 - self.p))

    def display(self):
        return 'Berboulli(p = %.2f)' % self.p

    def sample(self):
        return ((np.random.random() < self.p) and 1.0 or 0.0)

    # Note that this places probability mass outside of {0, 1}
    def density(self, xs):
        return (self.p * xs + (1.0 - self.p) * (1.0 - xs))

class Laplace:
    def __init__(self, mu = 0, b = 1, max_b = np.Inf):
        self.mu, self.b, self.max_b = mu, b, max_b
    
    def from_data(self, data, weights = None):
        m = np.average(data, weights = weights)
        v = np.average((data - m)**2, weights = weights)
        b = min(np.sqrt(v / 2), self.max_b)
        self.mu, self.b = m, b

    def mean(self):
        return self.mu

    def sd(self):
        return np.sqrt(2) * self.b

    def display(self):
        return 'Laplace(mu = %.2f, b = %.2f)' % (self.mu, self.b)

    def sample(self):
        return np.random.laplace(self.mu, self.b)

    def density(self, xs):
        c = 1 / (2 * self.b)
        return c * np.exp(-abs(xs - self.mu) / self.b)

class MultivariateNormal:
    def __init__(self, m = np.array([0]), c = np.matrix([[1]])):
        self.m, self.c = m, c
        self.k = len(m)

    def from_data(self, data, weights = None):
        n, k = data.shape[0], data.shape[1]
        m = np.average(data, axis = 0, weights = weights)
        data0 = data - m
        c = np.empty((k,k))
        for i in range(k):
            for j in range(k):
                if i < j: continue
                cell = np.average(data0[:,i] * data0[:,j], weights = weights)
                c[i,j] = cell
                if not i == j: c[j,i] = cell
        self.m, self.c, self.k = m, c, k

    def mean(self):
        return self.m

    def cov(self):
        return self.c

    def display(self):
        return 'MVNormal(mu = %s, Sigma = ...)' % str(self.m)

    def sample(self):
        return np.random.multivariate_normal(self.m, self.c)

    # TODO: very ugly, should clean up or at least explain...
    def density(self, xs):
        n = xs.shape[0]
        xs0 = xs - self.m
        c = (2.0 * np.pi) ** (-self.k / 2.0) * abs(la.det(self.c)) ** (-0.5)
        sigma_inv_sqrt = la.cholesky(la.inv(self.c))
        mdistsq = np.empty(n)
        for i in range(n):
            x_sigma_inv_sqrt = np.dot(xs0[i,], sigma_inv_sqrt)
            mdistsq[i] = np.dot(x_sigma_inv_sqrt, x_sigma_inv_sqrt)
        return c * np.exp(-0.5 * mdistsq)

class Normal:
    def __init__(self, m = 0, s = 1, max_sigma = np.Inf):
        self.m, self.s, self.max_s = m, s, max_sigma
    
    def from_data(self, data, weights = None):
        m = np.average(data, weights = weights)
        v = np.average((data - m)**2, weights = weights)
        s = min(np.sqrt(v), self.max_s)
        self.m, self.s = m, s

    def mean(self):
        return self.m

    def sd(self):
        return self.s

    def display(self):
        return 'Normal(mu = %.2f, sigma = %.2f)' % (self.m, self.s)

    def sample(self):
        return np.random.normal(self.m, self.s)

    def density(self, xs):
        return norm.pdf(xs, self.m, self.s)

class NormalFixedMean(Normal):
    def from_data(self, data, weights = None):
        v = np.average((data - self.m)**2, weights = weights)
        s = min(np.sqrt(v), self.max_s)
        self.s = s

class Kernel:
    def __init__(self, x = None, w = None, h = 1.0):
        self.x, self.w, self.h = x, w, h

    def from_data(self, data, weights = None):
        self.x, self.w = data, weights

    def mean(self):
        return np.average(self.x, weights = self.w)

    def sd(self):
        v = np.average((self.x - self.mean()) ** 2, weights = self.w)
        return np.sqrt(v + self.h_adapt() ** 2)

    def h_adapt(self):
        return self.h * max(1.0, np.sum(self.w)) ** (-0.2)

    def display(self):
        return 'Kernel density (n ~ %.2f, mean = %.2f, sd = %.2f; h = %.2f)' \
               % (np.sum(self.w), self.mean(), self.sd(), self.h_adapt())

    def density(self, xs):
        h = self.h_adapt()
        c = 1 / (h * np.sqrt(2 * np.pi))
        def d_single(x):
            contrib = np.exp(-0.5 * (x - self.x) ** 2 / (h ** 2))
            return np.average(contrib, weights = self.w)
        return c * (np.vectorize(d_single))(xs)
