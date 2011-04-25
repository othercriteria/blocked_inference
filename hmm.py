#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Splitting off HMM code.
# Daniel Klein, 4/25/2011

import numpy as np
import random

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
