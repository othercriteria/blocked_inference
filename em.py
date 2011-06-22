#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Daniel Klein, 2/28/2011

import numpy as np
from numpy.linalg import norm

def normalize(x):
    return x / sum(x)

def kmeans(data, K, epsilon = 0.01, max_reps = 50):
    data = np.array(data)
    num_data = len(data)
    data_mean, data_sd = np.mean(data), np.std(data)

    means = np.random.normal(data_mean, data_sd, K)

    J = np.Inf
    reps = 0
    while (reps < max_reps) and (J > epsilon * num_data):
        dists = np.empty((num_data, K))
        for k in range(K):
            dists[:,k] = (data - means[k]) ** 2
        best = np.argmin(dists, 1)

        for k in range(K):
            means[k] = np.mean(data[best == k])
        means[np.isnan(means)] = np.random.normal(data_mean, data_sd, K)

        J = np.sum((data - means[best]) ** 2)
        print J
        reps += 1

    return means

def em(data, num_class, dist, epsilon = 0.01, init_reps = 0, max_reps = 50,
       blocks = None, count_restart = 5.0, gamma_seed = None,
       true_gamma = None):
    data = np.array(data)
    num_data = data.shape[0]
    classes = range(num_class)
    if blocks is None:
        blocks = [np.arange(num_data)]
    num_block = len(blocks)

    # Initialize (blocked) mixing parametes
    pi_hat = np.array([normalize(np.array([1.0 for i in classes]))
                       for b in range(num_block)])

    # Initialize responsiblities, winner take all
    if not true_gamma is None:
        # Initialize with true class membership
        gamma_hat = np.zeros((num_class, num_data))
        for j, c in enumerate(true_gamma):
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
    dists_hat = [dist.from_data(data, gamma_hat[c]) for c in classes]

    reps = 0
    while True:
        # E step: from dists and pi, learn gamma
        phi_hat = np.array([(d.density())(data) for d in dists_hat])
        for b, block in enumerate(blocks):
            for c in classes:
                phi_hat[c,block] *= pi_hat[b,c]
        gamma_new = phi_hat / np.sum(phi_hat, 0)

        # M step: from gamma, learn dists and pi
        dists_new = []
        for c in classes:
            if sum(gamma_new[c]) < count_restart:
                idx = np.random.random_integers(num_data,
                                                size = int(count_restart)) - 1
                dists_new.append(dist.from_data(data[idx]))
            else:
                dists_new.append(dist.from_data(data, gamma_new[c]))
        pi_new = np.array([normalize(np.sum(gamma_new[:,block], 1))
                           for block in blocks])

        reps += 1
        converged = np.max(np.abs(gamma_hat - gamma_new)) < epsilon
        pi_hat, gamma_hat, dists_hat = pi_new, gamma_new, dists_new
        if (converged and reps >= init_reps) or (reps >= max_reps):
            break

    return { 'pi': pi_new,
             'dists': dists_new,
             'gamma': gamma_new,
             'reps': reps,
             'converged': converged }

