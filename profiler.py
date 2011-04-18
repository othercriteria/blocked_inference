#!/usr/bin/env python

# Experiments with EM estimation on HMM data. Profiling method performance.
# Daniel Klein, 3/31

import cProfile
import pstats

from test import *
cProfile.run('main()', 'hmmprof')

p = pstats.Stats('hmmprof')
p.strip_dirs().sort_stats('cumulative').print_stats(10)
