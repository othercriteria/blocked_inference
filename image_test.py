#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Testing application to image denoising.
# Daniel Klein, 4/18/2011

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from distributions import NormalFixedMean
from em import em
from visualization import display_hist, display_densities

# Parameters
image_file = 'broadway.jpg'
image_rescale = 15
noise_sd = 35.0
block_splits = 5
count_restart = 10.0
max_sigma = 60.0

# Load image
im = Image.open(image_file).convert('L')
width, height = im.size

# Resize image
width, height = int(width / image_rescale), int(height / image_rescale)
im = im.resize((width, height))

# Summary image
summary = Image.new('L', (width * 2 + 40, height * 2 + 60), 255)
draw = ImageDraw.Draw(summary)
draw.text((5, height + 10), 'Original')
draw.text((width + 25, height + 10), 'Noise SD = %.2f' % noise_sd)
draw.text((5, 2 * height + 40), 'Argmax')
draw.text((width + 25, 2 * height + 40), 'Average')
del draw
summary.paste(im, (10, 10))

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
noise = np.random.normal(0, noise_sd, width * height)
noisy_emissions = real_emissions + noise

# Generate noisy image
noisy = Image.new('L', (width, height))
noisy.putdata(noisy_emissions)
summary.paste(noisy, (30 + width, 10))

# Do EM
results = em(noisy_emissions,
             [NormalFixedMean(m, max_sigma = max_sigma) for m in range(256)],
             count_restart = count_restart,
             blocks = blocks)
dists = results['dists']
pi = results['pi']
print 'Iterations: %(reps)d' % results

gamma = np.transpose(results['gamma'])
means = np.array([d.mean() for d in dists])
sds = np.array([d.sd() for d in dists])

# Display summary figures
display_densities(real_emissions, dists)

# Reconstruct with argmax
im_argmax = Image.new('L', (width, height))
reconstruct_argmax = means[np.argmax(gamma, axis=1)]
im_argmax.putdata(reconstruct_argmax)
summary.paste(im_argmax, (10, 40 + height))

# Reconstruct with weighted average
im_avg = Image.new('L', (width, height))
reconstruct_avg = [np.average(means, weights=g, axis=0) for g in gamma]
im_avg.putdata(reconstruct_avg)
summary.paste(im_avg, (30 + width, 40 + height))

# Show summary image
summary.show()

# Display estimated per-level noise parameters
plt.figure()
plt.hist(sds, normed = True, bins = 20)
plt.show()

# Compare RMSE between reconstructions
def rmse(x):
    return np.sqrt(np.mean((x - real_emissions) ** 2))
print 'Raw MSE: %.1f' % rmse(noisy_emissions)
print 'ArgMax MSE: %.1f' % rmse(reconstruct_argmax)
print 'Avg MSE: %.1f' % rmse(reconstruct_avg)
