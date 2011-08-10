#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin, fmin_l_bfgs_b

num_data = 100
p = [0.2, 0.1]
mu = [0, 3, 6]

def fill(x):
    return np.array(list(x) + [1 - sum(x)])

p_full = fill(p)
ps = np.cumsum(p_full)

x = np.empty((num_data, 1))
for i in range(num_data):
    r = np.random.random()
    for this_p, m in zip(ps, mu):
        if r < this_p:
            x[i] = np.random.normal(m, 1)
            break

def dens(x, mu):
    return np.transpose(np.array((1 / np.sqrt(2 * np.pi)) *
                                  np.exp(-(x - mu) ** 2 / 2)))
phi = np.empty((num_data, len(mu)))
for c, m in enumerate(mu):
    phi[:,c] = dens(x, m)

def nll(p):
    p_full = np.array(list(p) + [1 - np.sum(p)])
    if np.min(p_full) < 0 or np.max(p_full) > 1:
        return np.Inf
    return -np.sum(np.log(np.dot(phi, p_full)))

def nll_prime(p):
    p_full = fill(p)
    return (-np.dot(np.transpose(phi)[0:2,:], 1 / np.dot(phi, p_full)) +
            np.dot(np.transpose(phi)[2,:], 1 / np.dot(phi, p_full)))

print nll([1.0/3, 1.0/3])
print nll(p)
print nll_prime(p)

opt_simplex = fmin(func = nll, x0 = np.array([1.0/3, 1.0/3]))
print opt_simplex
p_simplex = fill(opt_simplex).reshape((3,1))

opt_bfgs = fmin_l_bfgs_b(func = nll, x0 = np.array([1.0/3, 1.0/3]),
                         approx_grad = True,
                         bounds = [(0,1), (0,1)])
print opt_bfgs
p_bfgs = fill(opt_bfgs[0]).reshape((3,1))

opt_bfgs_grad = fmin_l_bfgs_b(func = nll, x0 = np.array([1.0/3, 1.0/3]),
                              fprime = nll_prime,
                              bounds = [(0,1), (0,1)])
print opt_bfgs_grad
p_bfgs_grad = fill(opt_bfgs_grad[0]).reshape((3,1))

phi = np.matrix(phi)
phi_t = np.transpose(phi)
hat = la.inv(phi_t * phi) * phi_t
unhat = la.inv(phi * phi_t) * phi
print np.dot(hat, 1 / (num_data * np.dot(unhat, np.ones((3, 1)))))
p_full = np.array(p_full)
p_full = p_full.reshape((3,1))
print np.dot(phi_t, 1 / np.dot(phi, p_bfgs_grad))
