#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Splitting off visualization routines from individual tests.
# Daniel Klein, 4/18/2011

import matplotlib.pyplot as plt
import numpy as np

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
