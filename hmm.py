#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Daniel Klein, 2/28/2011

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

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

def normalize(x):
    return x / sum(x)

def em(data, num_class, dist, epsilon = 0.0001, init_reps = 0, max_reps = 50,
       num_block = 1, count_restart = 5.0, gamma_seed = None):
    data = np.array(data)
    num_data = data.shape[0]
    classes = range(num_class)
    blocks = np.array_split(np.arange(num_data), num_block)

    # Initialize (blocked) mixing parametes
    pi_hat = np.array([normalize(np.array([1.0 for i in classes]))
                       for b in range(num_block)])

    # Initialize responsiblities, winner take all
    if not gamma_seed is None:
        old_state = np.random.get_state()
        np.random.seed(gamma_seed)
    r = np.random.multinomial(1, pi_hat[0], num_data)
    if not gamma_seed is None:
        np.random.set_state(old_state)
    order = np.argsort(data)
    gamma_hat = np.zeros((num_class, num_data))
    for i, o in enumerate(order):
        gamma_hat[:,o] = np.transpose(r[i])

    # Learn initial class-conditional distributions from data
    dists_hat = [dist.from_data(data, gamma_hat[i]) for i in classes]

    reps = 0
    while True:
        # E step
        phi_hat = np.array([map(d.density(), data) for d in dists_hat])
        for b, block in enumerate(blocks):
            for c in classes:
                phi_hat[c,block] *= pi_hat[b,c]
        gamma_new = phi_hat / np.sum(phi_hat, 0)

        # M step
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
        converged = np.min(np.abs(gamma_hat - gamma_new)) < epsilon
        pi_hat, gammma_hat, dists_hat = pi_new, gamma_new, dists_new
        if (converged and reps >= init_reps) or (reps >= max_reps):
            break

    return pi_new, dists_new, reps, converged

def print_mixture(pi, dists):
    for p, d in zip(np.transpose(pi), dists):
        print '%s: %s' % (p, d.display())
    print

if __name__ == '__main__':
    run_data = {}
    run_id = 0
    
    emissions_normal = { 1: Normal(0,   0.6),
                         2: Normal(3.5, 0.8),
                         3: Normal(6.5, 0.4) }
    emissions_laplace = { 1: Laplace(0, 0.6),
                          2: Laplace(3.5, 0.8),
                          3: Laplace(6.5, 0.4) }
    emission_spec = emissions_laplace
    dist = Laplace(max_b = 1.0) # Kernel(h = 0.3) # Laplace(max_b = 0.5)
    num_classes_guess = 3
    graphics_on = False
    num_state_reps = 5
    num_emission_reps = 2
    num_gamma_init_reps = 2

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
            if num_data < 2000 and num_data > 200: break

        # Generate shuffled indices for repeatable shuffling
        shuffling = np.arange(num_data)
        np.random.shuffle(shuffling)
        
        for emission_rep in range(num_emission_reps):
            print 'Emission repetition %d' % emission_rep
            model.emit()

            for shuffled in [False, True]:
                print 'Shuffling states/emissions: %s' % str(shuffled)
                states = np.array(model.state_vec)
                emissions = np.array(model.emission_vec)
                if shuffled:
                    states = states[shuffling]
                    emissions = emissions[shuffling]
                
                for num_block in [1, 2, 5, 10, 20, 50]:
                    print 'Blocks: %d' % num_block
                    for gamma_rep in range(num_gamma_init_reps):
                        print 'Initial gamma seed: %d' % gamma_rep

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
                                                   count_restart = 0.0)
                        run_time = time.clock() - start_time
                        this_run['run time'] = run_time
                        this_run['reps'] = reps

                        conv_status = conv and 'converged' or 'not converged'
                        this_run['convergence'] = conv_status

                        print 'Reps: %d (%s)' % (reps, conv_status)
                        print 'Time elapsed: %.2f' % run_time
                        print_mixture(pi, dists)

                        run_data[run_id] = this_run

    # Output data to CSV
    cols = set()
    for id in run_data:
        for k in run_data[id]:
            cols.add(k)
    with open('outfile.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(list(cols))
        writer.writerows([[run_data[id][c] for c in cols] for id in run_data])
    
