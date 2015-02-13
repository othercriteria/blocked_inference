#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Daniel Klein, 2/28/2011

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy

def normalize(x):
    return x / np.sum(x)

def kmeans(data, K, epsilon = 0.01, max_reps = 10):
    data = np.array(data)
    num_data, dim = data.shape[0], data.shape[1]
    data_mean, data_cov = np.mean(data, axis = 0), np.cov(data, rowvar = 0)
    if dim == 1:
        def make_means(n):
            return np.random.normal(data_mean, np.sqrt(data_cov), (n,1))
    else:
        def make_means(n):
            return np.random.multivariate_normal(data_mean, data_cov, n)

    means = make_means(K)

    J = np.Inf
    reps = 0
    while (reps < max_reps) and (J > epsilon * num_data):
        J_old = J
        dists = cdist(data, means)
        best = np.argmin(dists, axis = 1)

        for k in range(K):
            means[k] = np.mean(data[best == k], axis = 0)
            if np.isnan(means[k,0]):
                means[k] = make_means(1)

        J = 0
        for i in range(num_data):
            J += norm(data[i] - means[best[i]])
        if J == J_old: break
        reps += 1

    return { 'means': means, 'best': best }

def pi_maximize(data, dists):
    num_data = data.shape[0]
    num_comps = len(dists)
    
    phi = np.empty((num_data, num_comps))
    for c in range(num_comps):
        phi[:,c] = dists[c].density(data)
    phi = np.matrix(phi)
    phi_t = np.transpose(phi)
    def fill(p):
        return np.array(list(p) + [1 - np.sum(p)]).reshape((num_comps, 1))
    def nll(p):
        p_full = fill(p)
        if np.min(p_full) < 0 or np.max(p_full) > 1:
            return np.Inf
        return -np.sum(np.log(np.dot(phi, p_full)))
    def nll_prime(p):
        p_full = fill(p)
        grad = np.dot(-phi_t[0:(num_comps-1),:] + phi_t[(num_comps-1),:],
                      1 / np.dot(phi, p_full))
        return np.array(grad, order = 'F')
    opt = fmin_l_bfgs_b(func = nll, fprime = nll_prime,
                        x0 = np.array([1.0 / num_comps] * (num_comps-1)),
                        bounds = [(0, 1)] * (num_comps-1))
    return fill(opt[0])

def em(data, dists, epsilon = 0.01, init_reps = 0, max_reps = 50,
       blocks = None, count_restart = 0.0, gamma_seed = None,
       init_gamma = None, trace = False, pi_max = False):
    data = np.array(data)
    num_data = data.shape[0]
    num_class = len(dists)
    classes = range(num_class)
    if blocks is None:
        blocks = [np.arange(num_data)]
    num_block = len(blocks)

    # Initialize (blocked) mixing parameters
    pi_hat = np.array([normalize(np.array([1.0 for i in classes]))
                       for b in range(num_block)])

    # Initialize responsiblities, winner take all
    if not init_gamma is None:
        # Initialize with true class membership
        gamma_hat = np.zeros((num_class, num_data))
        for j, c in enumerate(init_gamma):
            gamma_hat[c,j] = 1.0
    else:
        # Random initialization
        if not gamma_seed is None:
            old_state = np.random.get_state()
            np.random.seed(gamma_seed)
        r = np.random.multinomial(1, pi_hat[0], num_data)
        if not gamma_seed is None:
            np.random.set_state(old_state)
        order = np.argsort(map(str,data))
        gamma_hat = np.empty((num_class, num_data))
        for i, o in enumerate(order):
            gamma_hat[:,o] = np.transpose(r[i])

    # Learn initial class-conditional distributions from data
    for c in classes:
        dists[c].from_data(data, gamma_hat[c])

    reps = 0
    phi_hat = np.empty((num_class, num_data))
    if trace:
        dists_trace = [deepcopy(dists)]
        pi_trace = [deepcopy(pi_hat)]
    while True:
        # E step: from dists and pi, learn gamma
        for c in classes:
            phi_hat[c] = dists[c].density(data)
        for b, block in enumerate(blocks):
            for c in classes:
                phi_hat[c,block] *= pi_hat[b,c]
        gamma_new = phi_hat / np.sum(phi_hat, 0)

        # M step: from gamma, learn dists and pi
        for c in classes:
            if np.sum(gamma_new[c]) < count_restart:
                idx = np.random.random_integers(num_data,
                                                size = int(count_restart)) - 1
                dists[c].from_data(data[idx])
            else:
                dists[c].from_data(data, gamma_new[c])
        if trace:
            dists_trace.append(deepcopy(dists))
        if pi_max:
            for b, block in enumerate(blocks):
                pi_hat[b] = np.transpose(pi_maximize(data[block], dists))
        else:
            for b, block in enumerate(blocks):
                pi_hat[b] = normalize(np.sum(gamma_new[:,block], 1))
        if trace:
            pi_trace.append(deepcopy(pi_hat))

        reps += 1
        converged = np.max(np.abs(gamma_hat - gamma_new)) < epsilon
        gamma_hat = gamma_new
        if (converged and reps >= init_reps) or (reps >= max_reps):
            break

    results = { 'pi': pi_hat,
                'dists': dists,
                'gamma': gamma_hat,
                'reps': reps,
                'converged': converged }
    if trace:
        results['dists_trace'] = dists_trace
        results['pi_trace'] = pi_trace
    return results

