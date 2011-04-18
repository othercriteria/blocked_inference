#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Testing application to image denoising.
# Daniel Klein, 4/18/2011

from PIL import Image
import numpy as np

from distributions import Normal
from em import em
from visualization import display_hist, display_densities

# Parameters
image_file = 'broadway.jpg'
image_rescale = 4
noise_sd = 8.0
num_components = 12
comp_dist = Normal(max_sigma = 20.0)
block_splits = 5
count_restart = 50.0

# Load image
im = Image.open(image_file).convert('L')
width, height = im.size

# Resize image
width, height = int(width / image_rescale), int(height / image_rescale)
im = im.resize((width, height))

# Flatten to emissions
real_emissions = list(im.getdata())
num_data = len(real_emissions)
real_emissions = np.array(real_emissions)

# Block emissions
width_blocks = np.array_split(np.arange(width), block_splits)
height_blocks = np.array_split(np.arange(height), block_splits)
idx = np.arange(num_data)
idx.resize((height, width))
blocks = []
for hb in height_blocks:
    for wb in width_blocks:
        block = [idx[h, w] for h in hb for w in wb]
        blocks.append(np.array(block))

# Generate noise
noise_dat = np.random.normal(0, noise_sd, width * height)

# Generate noisy image
noisy = Image.new('L', (width, height))
noisy.putdata(real_emissions + noise_dat)
noisy_emissions = list(noisy.getdata())
noisy.show()

# Do EM
results = em(noisy_emissions,
             num_components,
             comp_dist,
             count_restart = count_restart,
             smart_gamma = False)
dists = results['dists']
pi = results['pi']
gamma = np.transpose(results['gamma'])
means = np.array([d.mean() for d in dists])

# Display summary figures
# display_densities(real_emissions, dists)
# display_hist(real_emissions, dists)

# Reconstruct with argmax
im_argmax = Image.new('L', (width, height))
im_argmax.putdata(means[np.argmax(gamma, axis=1)])
im_argmax.show()

# Reconstruct with weighted average
im_avg = Image.new('L', (width, height))
im_avg.putdata([np.average(means, weights=g, axis=0) for g in gamma])
im_avg.show()
