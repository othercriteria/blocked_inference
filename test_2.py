#!/usr/bin/env python

# Simpler test for blocking, using permutation to weaken block structure.
# Daniel Klein, 7/15/2011

import numpy as np
import random

from em import kmeans, em
from distributions import Normal, NormalFixedVariance, Kernel
from visualization import display_densities, display_hist

# Parameters
reps = 1
em_steps = 1000
mu = np.array([0.175, -0.525])
p = 0.75
num_blocks = 2
n_block = 2000
block_bias_sd = 0.1
count_restart = 0.0
model = [NormalFixedVariance(s = 1.0) for n in range(2)]
init = 'kmeans' # Choices: random, kmeans, true
graphics = False

# Set random seeds for reproducible runs
random.seed(163)
np.random.seed(137)

for rep in range(reps):
    init_gamma = None
    
    # Generate block mixing parameters that preserve the global mean
    while True:
        block_bias = np.random.normal(0, block_bias_sd, num_blocks)
        block_bias = block_bias - np.mean(block_bias)
        block_bias = block_bias * (block_bias_sd / np.sqrt(np.var(block_bias)))
        block_p = p + block_bias
        if (block_p > 0).all() and (block_p < 1).all(): break
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
                 max_reps = em_steps)
    print 'Iterations: %(reps)d' % results
    dists = results['dists']
    pi = results['pi']

    # Display results
    for p, d in zip(np.transpose(pi), dists):
        print '%s: %s' % (p, d.display())
    print
    if graphics:
        display_densities(data, dists)
        display_hist(data, dists)

