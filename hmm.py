#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Daniel Klein, 2/28/2011

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from distributions import Normal, Laplace, Kernel

# Set random seeds for reproducible runs.
random.seed(163)
np.random.seed(137)

# Adapted from:
# http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
class RandomGenerator(object):
    def __init__(self, vals, weights):
        self.vals = vals
        self.totals = np.cumsum(weights)

    def next(self):
        rnd = random.random() * self.totals[-1]
        ind = np.searchsorted(self.totals, rnd)
        return self.vals[ind]

    def __call__(self):
        return self.next()

# Distinguished states 'Start' and 'End' without emissions
class HMM:
    def __init__(self, transitions, emissions):
        self.states = set()
        self.transition_matrix = {}
        self.transitions = {}
        for state, new_states, weights in transitions:
            self.states.add(state)
            self.transitions[state] = RandomGenerator(new_states, weights)
            for s, w in zip(new_states, weights):
                self.transition_matrix[(state, s)] = w
        self.states.difference_update(['Start', 'End'])

        self.emissions = emissions

    def simulate(self):
        self.state_vec = []
        state = 'Start'
        while True:
            state = self.transitions[state]()
            if state == 'End': break
            self.state_vec.append(state)

    def emit(self):
        self.emission_vec = [self.emissions[state].sample()
                             for state in self.state_vec]

def normalize(x):
    return x / sum(x)

def em(data, num_class, dist, epsilon = 0.01, init_reps = 0, max_reps = 50,
       num_block = 1, count_restart = 5.0, gamma_seed = None,
       smart_gamma = True, true_gamma = None):
    data = np.array(data)
    num_data = data.shape[0]
    classes = range(num_class)
    blocks = np.array_split(np.arange(num_data), num_block)

    # Initialize (blocked) mixing parametes
    pi_hat = np.array([normalize(np.array([1.0 for i in classes]))
                       for b in range(num_block)])

    # Initialize responsiblities, winner take all
    if not true_gamma is None:
        # Initialize with true class membership
        # This is sort of a hack in how it handles mapping of states to
        # entries in gamma_hat.
        gamma_hat = np.zeros((num_class, num_data))
        for j, c in enumerate(true_gamma):
            gamma_hat[c-1,j] = 1.0
    elif smart_gamma:
        # Data dependent
        breaks = np.linspace(0, 100, num_class + 1)[1:]
        quantile = np.percentile(data, list(breaks))
        gamma_hat = np.zeros((num_class, num_data))
        for j in range(num_data):
            for i in classes:
                if data[j] <= quantile[i]:
                    break
            gamma_hat[i,j] = 1.0
    else:
        # Random initialization
        if not gamma_seed is None:
            old_state = np.random.get_state()
            np.random.seed(gamma_seed)
        r = np.random.multinomial(1, pi_hat[0], num_data)
        if not gamma_seed is None:
            np.random.set_state(old_state)
        order = np.argsort(data)
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
                subsample = random.sample(data, int(count_restart))
                dists_new.append(dist.from_data(subsample))
            else:
                dists_new.append(dist.from_data(data, gamma_new[c]))
        pi_new = np.array([normalize(np.sum(gamma_new[:,block], 1))
                           for block in blocks])

        reps += 1
        converged = np.max(np.abs(gamma_hat - gamma_new)) < epsilon
        pi_hat, gamma_hat, dists_hat = pi_new, gamma_new, dists_new
        if (converged and reps >= init_reps) or (reps >= max_reps):
            break

    return pi_new, dists_new, reps, converged

# To be used in assessing model performance
# I'm dismayed at how slowly the recursive approach works in Python
def permute(list):
    if len(list) == 1:
        return [list]
    else:
        p = []
        for i, x in enumerate(list):
            rest = list[:i] + list[(i+1):]
            for pr in permute(rest):
                p.append([x] + pr)
        return p

def print_mixture(pi, dists):
    for p, d in zip(np.transpose(pi), dists):
        print '%s: %s' % (p, d.display())
    print

def max_error_mean(dists, dists_target):
    return min([max([abs(d.mean() - dt.mean())
                     for d, dt in zip(dists_perm, dists_target)])
                for dists_perm in permute(dists)])

def mean_error_mean(dists, dists_target):
    return min([np.mean([abs(d.mean() - dt.mean())
                         for d, dt in zip(dists_perm, dists_target)])
                for dists_perm in permute(dists)])

def display_hist(data, distributions):
    plt.figure()
    plt.hist(data, normed = True, bins = 30)
    for d in distributions:
        mu, sigma = d.mean(), d.sd()
        plt.axvline(mu, linewidth=2)
        plt.axvline(mu - 2 * sigma, linestyle='--')
        plt.axvline(mu + 2 * sigma, linestyle='--')
    plt.show()

def display_densities(data, distributions):
    points = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)
    plt.figure()
    for d in distributions:
        plt.plot(points, (d.density())(points))
    plt.show()
    
def main():
    run_data = {}
    run_id = 0

    scale = 0.5
    emissions_normal = { 1: Normal(0,   2.0 * scale),
                         2: Normal(3.5, 3.0 * scale),
                         3: Normal(6.5, 1.0 * scale) }
    emissions_laplace = { 1: Laplace(0, 2.0 * scale),
                          2: Laplace(3.5, 3.0 * scale),
                          3: Laplace(6.5, 1.0 * scale) }
    emission_spec = emissions_normal
    dist = Normal(max_sigma = 6.0)
    num_classes_guess = 3
    num_state_reps = 10
    num_emission_reps = 2
    num_gamma_init_reps = 1
    num_blocks = [1, 2, 5, 10, 20, 50, 100]
    verbose = False
    graphics_on = False

    total_work = (num_state_reps * num_emission_reps *
                  2 * num_gamma_init_reps * len(num_blocks))

    work = 0
    for state_rep in range(num_state_reps):
        print 'State repetition %d' % state_rep

        # Generate HMM states
        while True:
            model = HMM([('Start', (1,),          (1.0,)),
                         (1,       (1,2,3),       (0.98, 0.02, 0.0)),
                         (2,       (1,2,3),       (0.02, 0.95,  0.03)),
                         (3,       (1,2,3,'End'), (0.03,  0.03,  0.93, 0.01))],
                    emission_spec)
            model.simulate()
            num_data = len(model.state_vec)
            if num_data < 4000 and num_data > 200: break

        counts = {}
        for state in model.state_vec:
            if not state in counts:
                counts[state] = 0
            counts[state] += 1
        if verbose: print 'Counts: %s' % str(counts)

        # Generate shuffled indices for repeatable shuffling
        shuffling = np.arange(num_data)
        np.random.shuffle(shuffling)
        
        for emission_rep in range(num_emission_reps):
            if verbose: print 'Emission repetition %d' % emission_rep
            model.emit()

            for shuffled in [False, True]:
                if verbose: print 'Shuffling HMM run: %s' % str(shuffled)
                states = np.array(model.state_vec)
                emissions = np.array(model.emission_vec)
                if shuffled:
                    states = states[shuffling]
                    emissions = emissions[shuffling]
                
                for num_block in num_blocks:
                    if verbose: print 'Blocks: %d' % num_block
                    for gamma_rep in range(num_gamma_init_reps):
                        if verbose: print 'Initial gamma seed: %d' % gamma_rep

                        run_id += 1
                        this_run = {}

                        this_run['num data'] = num_data
                        this_run['state rep'] = state_rep
                        this_run['emission rep'] = emission_rep
                        this_run['shuffled'] = shuffled
                        this_run['blocks'] = num_block
                        this_run['gamma init rep'] = gamma_rep

                        start_time = time.clock()
                        pi, dists, reps, conv = em(emissions,
                                                   num_classes_guess,
                                                   dist,
                                                   num_block = num_block,
                                                   gamma_seed = gamma_rep,
                                                   smart_gamma = False,
                                                   true_gamma = states,
                                                   count_restart = 0.0)
                        run_time = time.clock() - start_time
                        this_run['run time'] = run_time
                        this_run['reps'] = reps

                        conv_status = conv and 'converged' or 'not converged'
                        this_run['convergence'] = conv_status

                        print 'Reps: %d (%s)' % (reps, conv_status)
                        print 'Time elapsed: %.2f' % run_time
                        if verbose: print_mixture(pi, dists)

                        if graphics_on:
                            display_densities(emissions, dists)
                            display_hist(emissions, dists)

                        act = emission_spec.values()
                        this_run['err mean max'] = max_error_mean(dists, act)
                        this_run['err mean mean'] = mean_error_mean(dists, act)

                        like = np.zeros(num_data)
                        pi_overall = np.mean(pi, 0)
                        for p, dist in zip(pi_overall, dists):
                            like += p * (dist.density())(states)
                        this_run['log likelihood'] = np.sum(np.log(like))

                        run_data[run_id] = this_run

                        work += 1
                        print 'Finished run %d/%d' % (work, total_work)

    # Output data to CSV
    cols = set()
    for id in run_data:
        for k in run_data[id]:
            cols.add(k)
    with open('outfile.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(list(cols))
        writer.writerows([[run_data[id][c] for c in cols] for id in run_data])

if __name__ == '__main__':
    main()
