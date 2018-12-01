#!/usr/bin/env python

import gpustat

stats = gpustat.GPUStatCollection.new_query()

for gpu in stats:
    print(gpu)
