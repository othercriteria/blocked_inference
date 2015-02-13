#!/usr/bin/env python

# Simpler test for blocking, using permutation to weaken block structure.
# Daniel Klein, 7/15/2011

import numpy as np
import matplotlib.pyplot as plt
import random

from em import kmeans, em
from distributions import Normal, NormalFixedVariance, Kernel
from visualization import display_densities, display_hist

# Parameters
reps = 1
em_steps = 300
pi_max = False
mu = np.array([0.175, -0.525])
p = 0.75
num_blocks = 2
n_block = 10000
block_bias_sd = 0.2
count_restart = 0.0
model = [NormalFixedVariance(s = 1.0) for n in range(2)]
init = 'kmeans' # Choices: random, kmeans, true
graphics = False
show_each = True
plot_trace = True


# Convenience function
def rmse(dists):
    m1, m2 = dists[0].mean(), dists[1].mean()
    error = np.sqrt(min([np.mean((np.array([m1,m2]) - mu) ** 2),
                         np.mean((np.array([m2,m1]) - mu) ** 2)]))
    return error

# Set random seeds for reproducible runs
random.seed(163)
np.random.seed(137)

errors = []
for rep in range(reps):
    if not show_each:
        print '.',
    init_gamma = None
    
    # Generate block mixing parameters that preserve the global mean
    while True:
        if num_blocks == 1:
            block_p = np.array([p])
            break
        block_bias = np.random.normal(0, block_bias_sd, num_blocks)
        block_bias = block_bias - np.mean(block_bias)
        block_bias = block_bias * (block_bias_sd / np.sqrt(np.var(block_bias)))
        block_p = p + block_bias
        if (block_p > 0).all() and (block_p < 1).all(): break
    if show_each:
        print 'Attained block bias SD: %.2f' % np.sqrt(np.var(block_p))

    # Generate data
    n = num_blocks * n_block
    data_p = []
    for i in range(num_blocks):
        data_p += [block_p[i]] * n_block
    data_p = np.array(data_p)
    data_comp = np.zeros(n, dtype=int)
    data_comp[np.random.sample(n) > data_p] = 1
    if init == 'true':
        init_gamma = data_comp
    data_mu = mu[data_comp]
    data = np.random.normal(data_mu, 1)
    blocks = np.array_split(np.arange(n), num_blocks)

    # Initialize with K-means
    if init == 'kmeans':
        init_gamma = kmeans(data.reshape((n,1)), 2)['best']

    # Do EM
    results = em(data,
                 model,
                 count_restart = count_restart,
                 blocks = blocks,
                 init_gamma = init_gamma,
                 init_reps = em_steps,
                 max_reps = em_steps,
                 pi_max = pi_max,
                 trace = True)
    if show_each:
        print 'Iterations: %(reps)d' % results
    dists, dists_trace = results['dists'], results['dists_trace']
    pi, pi_trace = results['pi'], results['pi_trace']

    # Display results
    if show_each:
        for p, d in zip(np.transpose(pi), dists):
            print '%s: %s' % (p, d.display())
        print
    if graphics:
        display_densities(data, dists)
        display_hist(data, dists)
    if plot_trace:
        pi_trace = np.array(pi_trace)
        v = np.transpose(pi_trace[:,:,0])

        u = np.empty((em_steps+1))
        w = np.empty((num_blocks, (em_steps+1)))
        m = np.empty((2, em_steps+1))
        for t in range((em_steps+1)):
            d = dists_trace[t]
            m1, m2 = d[0].mean(), d[1].mean()
            u[t] = m1 - m2
            m[0,t] = m1
            m[1,t] = m2
            for b in range(num_blocks):
                w[b,t] = v[b,t] * m1 + (1 - v[b,t]) * m2
            
        fig = plt.figure()
        for b in range(num_blocks):
            ax = fig.add_subplot(4, num_blocks, (b+1))
            plt.xlabel('u')
            plt.ylabel('v')
            plt.axis([-2.0, 2.0, 0.0, 1.0])
            plt.plot(u, v[b], '.')
        for b in range(num_blocks):
            ax = fig.add_subplot(4, num_blocks, (b+1) + num_blocks)
            plt.xlabel('step')
            plt.ylabel('v')
            plt.axis([0.0, em_steps, 0.0, 1.0])
            plt.plot(range((em_steps+1)), pi_trace[:,b,0])
        for b in range(num_blocks):
            ax = fig.add_subplot(4, num_blocks, (b+1) + num_blocks * 2)
            plt.xlabel('step')
            plt.ylabel('w')
            plt.axis([0.0, em_steps, -1.0, 1.0])
            plt.plot(range((em_steps+1)), w[b])
        for b in range(num_blocks):
            ax = fig.add_subplot(4, num_blocks, (b+1) + num_blocks * 3)
            plt.xlabel('step')
            plt.ylabel('means')
            plt.axis([0.0, em_steps, -2.0, 2.0])
            plt.plot(range((em_steps+1)), m[0])
            plt.plot(range((em_steps+1)), m[1])
        plt.savefig('test_2_traces.png')
        plt.show()

    # Compute error
    errors.append(rmse(dists))

# Summarize runs
errors = np.array(errors)
print '\nRMSE: %.2f +/- %.2f' % (np.mean(errors), 2.0 * np.std(errors))
