#!/usr/bin/env python

# Test of blocked EM for Ising model with unknown emission distribution.
# Daniel Klein, 7/15/2011

import numpy as np
from numpy import mod, exp
import matplotlib.pyplot as plt
import random

from em import kmeans, em
from distributions import Normal
from visualization import display_densities, display_hist

# Parameters
reps = 100
mus = np.linspace(0.0, 5.0, 40)
dim = 50
theta = [0.0, 1.0]
temp = 2.269 # T_c = 2.269?
gibbs_sweeps = 30
model = [Normal() for n in range(2)]
pi_max = True
count_restart = 0.0
init = 'kmeans' # Choices: random, kmeans, true
block_strategies = ['none', 'mean_4n', 'all_4n', 'mean_8n', 'all_8n', 'perfect']
graphics = False
show_each = False


# Set random seeds for reproducible runs
random.seed(163)
np.random.seed(137)

# Precalculate neighbors
neighbors_4 = {}
neighbors_8 = {}
for i in range(dim):
    for j in range(dim):
        i_m, i_p = mod(i-1, dim), mod(i+1, dim)
        j_m, j_p = mod(j-1, dim), mod(j+1, dim)
        neighbors_4[(i,j)] = [(i, j_m), (i, j_p),
                              (i_m, j), (i_p, j)]
        neighbors_8[(i,j)] = neighbors_4[(i,j)] + [(i_m, j_m), (i_m, j_p),
                                                   (i_p, j_m), (i_p, j_p)]

# Convenience function
def rmse(dists, true_mu):
    m1, m2 = dists[0].mean(), dists[1].mean()
    s1, s2 = dists[0].sd(), dists[1].sd()
    real = np.array([0.0, true_mu, 1.0, 1.0])
    error = np.sqrt(min([np.mean((np.array([m1,m2,s1,s2]) - real) ** 2),
                         np.mean((np.array([m2,m1,s2,s1]) - real) ** 2)]))
    return error

def block(data, block_strategy, true = None):
    blocks = None
    if block_strategy in ['mean_4n', 'mean_8n', 'all_4n', 'all_8n']:
        global_mean_loo = np.empty((dim,dim))
        global_sum = np.sum(data)
        n = dim * dim - 1.0
        for i in range(dim):
            for j in range(dim):
                loc = (i,j)
                global_mean_loo[loc] = (global_sum - data[loc]) / n
    if block_strategy in ['mean_4n', 'mean_8n']:
        if block_strategy == 'mean_4n':
            neighbors = neighbors_4
        if block_strategy == 'mean_8n':
            neighbors = neighbors_8
        nbr_mean = np.empty((dim,dim))
        for i in range(dim):
            for j in range(dim):
                loc = (i,j)
                nbr_mean[loc] = np.mean([data[l] for l in neighbors[loc]])
        block_greater = (nbr_mean > global_mean_loo).reshape((dim*dim,))
        indices = np.arange(dim*dim)
        blocks = [indices[block_greater], indices[-block_greater]]
    if block_strategy in ['all_4n', 'all_8n']:
        global_mean_loo = global_mean_loo.reshape((dim,dim,1))
        if block_strategy == 'all_4n':
            neighbors = neighbors_4
            global_mean_loo = np.repeat(global_mean_loo, 4, axis=2)
            nbr = np.empty((dim,dim,4))
        if block_strategy == 'all_8n':
            neighbors = neighbors_8
            global_mean_loo = np.repeat(global_mean_loo, 8, axis=2)
            nbr = np.empty((dim,dim,8))
        for i in range(dim):
            for j in range(dim):
                loc = (i,j)
                nbr[loc] = [data[l] for l in neighbors[loc]]
        block_g = (nbr > global_mean_loo).all(axis = 2).reshape((dim*dim,))
        block_l = (nbr < global_mean_loo).all(axis = 2).reshape((dim*dim,))
        block_o = -(block_g + block_l)
        indices = np.arange(dim*dim)
        blocks = [indices[block_g], indices[block_l], indices[block_o]]
    if block_strategy == 'perfect':
        true_vec = np.reshape(true, (dim*dim,))
        indices = np.arange(dim*dim)
        blocks = [indices[true_vec == -1], indices[true_vec == 1]]
    return blocks

errors = {}
for mu in mus:
    errors[mu] = {}
    for block_strategy in block_strategies:
        errors[mu][block_strategy] = []

# Vectorized evens and odds (for checkerboard updates)
r = np.empty((dim,dim),dtype=int)
c = np.empty((dim,dim),dtype=int)
for row in range(dim):
    r[row,:] = row
for col in range(dim):
    c[:,col] = col
evens = (((r + c) % 2) == 0)
odds = -evens

x_nbr_tot = np.empty((dim,dim),dtype=int)
for rep in range(reps):
    # Gibbs sampler for sampling hidden state
    beta = 1.0 / temp
    x = np.zeros((dim,dim),dtype=int) + 1
    for i in range(dim):
        for j in range(dim):
            if random.random() < 0.5:
                x[(i,j)] = -x[(i,j)]
    for sweep in range(gibbs_sweeps):
        r = np.random.random((dim,dim))
        for current in [evens, odds]:
            for i in range(dim):
                for j in range(dim):
                    loc = (i,j)
                    if not current[loc]: continue
                    x_nbr = [x[l] for l in neighbors_4[loc]]
                    x_nbr_tot[loc] = np.sum(x_nbr)
            delta_e = 2.0 * (theta[0] * x + theta[1] * (x * x_nbr_tot))
            p = np.exp(-beta * delta_e)
            flip = (delta_e < 0.0) + (r < p)
            x[current * flip] = -x[current * flip]

    if graphics:
        plt.imshow(x)
        plt.show()

    for mu in mus:
        init_gamma = None
        
        # Emissions according to mixture model
        data_comp = np.empty((dim,dim), dtype=int)
        data_comp[x == -1] = 0
        data_comp[x ==  1] = 1
        if init == 'true':
            init_gamma = data_comp
        data_mu = (np.array([0.0, mu]))[data_comp]
        data = np.random.normal(data_mu, 1)
        if graphics:
            plt.imshow(data)
            plt.show()

        # Initialize with K-means
        if init == 'kmeans':
            init_gamma = kmeans(data.reshape((dim*dim,1)), 2)['best']

        # Do (potentially adaptive) blocked EM, depending on strategy
        for block_strategy in block_strategies:
            # Only 'perfect' strategy uses the true states
            blocks = block(data, block_strategy, true = x)
        
            # Do EM
            results = em(data.reshape((dim*dim,)),
                         model,
                         count_restart = count_restart,
                         blocks = blocks,
                         init_gamma = init_gamma,
                         pi_max = pi_max)
            print 'Iterations: %d (%s)' % (results['reps'], block_strategy)
            dists = results['dists']
            pi = results['pi']

            # Display results
            if show_each:
                for p, d in zip(np.transpose(pi), dists):
                    print '%s: %s' % (p, d.display())
                print
            if graphics:
                display_densities(data.reshape((dim*dim,)), dists)
                display_hist(data.reshape((dim*dim,)), dists)

            # Compute errors
            errors[mu][block_strategy].append(rmse(dists, mu))

# Summarize runs
errs, confs = {}, {}
for block_strategy in block_strategies:
    errs[block_strategy], confs[block_strategy] = [], []
for mu in mus:
    print 'mu = %.2f' % mu
    for block_strategy in block_strategies:
        error = np.array(errors[mu][block_strategy])
        err, conf = np.mean(error), 2.0 * np.std(error) / np.sqrt(reps)
        errs[block_strategy].append(err)
        confs[block_strategy].append(conf)
        print 'RMSE (%s): %.2f +/- %.2f' % (block_strategy, err, conf)
    print

# Produce output for plotting
def vecify(x):
    return 'c(' + ','.join(map(str, x)) + ')'

print '# reps = %d, dim = %d, theta = [%.2f, %.2f], temp = %.3f, sweeps = %d' \
    % (reps, dim, theta[0], theta[1], temp, gibbs_sweeps)

print 'mu <- %s' % vecify(mus)
for block_strategy in block_strategies:
    name = 'rmse.' + (block_strategy.replace('_', '.'))
    print '%s <- %s' % (name, vecify(errs[block_strategy]))
    print '%s.conf <- %s' % (name, vecify(confs[block_strategy]))

print 'plot(c(0,%f), c(0,0.5), type="n", xlab="mu_1", ylab="RMSE")' % max(mus)
colors = ['black', 'red', 'blue', 'yellow', 'orange', 'cyan', 'magenta']
for block_strategy, color in zip(block_strategies, colors):
    name = 'rmse.' + (block_strategy.replace('_', '.'))
    print 'points(mu, %s, col="%s")' % (name, color)
    print 'segments(x0=mu, y0=%s-%s.conf, y1=%s+%s.conf)' % tuple([name] * 4)
