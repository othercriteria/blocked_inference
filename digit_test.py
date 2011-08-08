#!/usr/bin/env python

# Experiments with EM estimation on structured data.
# Daniel Klein, 4/25/2011

import numpy as np
from PIL import Image, ImageDraw

from distributions import Product, Bernoulli
from em import em
from process_digits_data import get_data

# Parameters
digits_of_interest = range(10) # [2, 3, 4] # range(10)
num_components = 10
num_blocks = [1, 3, 10, 30, 100, 300]
num_data = 600
init_to_labels = False

# Get data from beginning of MNIST data set
data = get_data(num_data)

# Extract digits of interest, keeping count of number of examples
image_counts = {}
for d in digits_of_interest:
    image_counts[d] = 0
emissions = []
labels = []
for l, i in data:
    if not l in digits_of_interest: continue
    emissions.append(np.array(i) / 255)
    labels.append(l)
    image_counts[l] += 1
print 'Number of examples:'
for d in image_counts:
    print '%d: %d' % (d, image_counts[d])
print

# Initialize summary image
summary = Image.new('L', (28 * num_components + 65, 28 * len(num_blocks)), 255)

# Do inference for varying numbers of blocks
idxs = np.argsort(map(np.sum, emissions))
reps = []
for block_i, num_block in enumerate(num_blocks):
    # Block data
    blocks = np.array_split(idxs, num_block)

    # Run EM
    results = em(emissions,
                 [Product([Bernoulli() for i in range(28 * 28)])
                  for n in range(num_components)],
                 count_restart = 3.0,
                 blocks = blocks,
                 gamma_seed = 137,
                 init_gamma = (init_to_labels and labels or None))
    dists = results['dists']
    print 'Reps: %d' % results['reps']
    reps.append(results['reps'])

    # Produce summary image
    offset = 0
    im = Image.new('L', (28 * len(dists), 28))
    for d in results['dists']:
        digit = Image.new('L', (28, 28))
        digit.putdata(np.array(d.mean()) * 255)
        im.paste(digit, (offset, 0))
        offset += 28
    summary.paste(im, (0, 28 * block_i))

# Annotate summary image
draw = ImageDraw.Draw(summary)
for i, (num_block, rep) in enumerate(zip(num_blocks, reps)):
    draw.text((28 * num_components + 5, 28 * i + 5), str(num_block))
    draw.text((28 * num_components + 35, 28 * i + 5), str(rep))
del draw
summary.save('digit_test_summary.png')
