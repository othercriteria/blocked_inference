#!/usr/bin/env python

# Experiments with EM estimation on structured data.
# Daniel Klein, 4/25/2011

import numpy as np
from PIL import Image

from distributions import Product, Bernoulli
from em import em
from process_digits_data import get_data

digits_of_interest = [2, 3, 4] # range(10)
num_components = 3
comp_dist = Product([Bernoulli() for i in range(28 * 28)])

data = get_data(500)

image_counts = {}
for d in digits_of_interest:
    image_counts[d] = 0

emissions = []
for l, i in data:
    if not l in digits_of_interest: continue
    emissions.append(np.array(i) / 255)
    image_counts[l] += 1

print 'Number of examples:'
for d in image_counts:
    print '%d: %d' % (d, image_counts[d])

results = em(emissions,
             num_components,
             comp_dist)
dists = results['dists']
print 'Reps: %d' % results['reps']

offset = 0
im = Image.new('L', (28 * len(dists), 28))
for d in results['dists']:
    digit = Image.new('L', (28, 28))
    digit.putdata(np.array(d.mean()) * 255)
    im.paste(digit, (offset, 0))
    offset += 28
im.show()
