#!/usr/bin/env python

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('default')

from argparse import ArgumentParser

parser = ArgumentParser(description='Compare ROC curves')
#parser.add_argument('input', nargs = '+', help="Paths to npy files")
parser.add_argument('--output', '-o', dest = 'output_name', help='Output file name (no extension)')

args = parser.parse_args()

models = [
    ('/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi/featured_3d_conv_roc.npy', 'Binning'),
    ('/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi/sparse_conv_single_neighbors_roc.npy', 'Single neighbors'),
    ('/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi/sparse_conv_hidden_aggregators_roc.npy', 'Hidden aggregators')
]

plt.figure(figsize=(6., 6.))

lw = 2

#for fname in args.input:
for fname, title in models:
    with open(fname, 'rb') as source:
        data = np.load(source)

    tp = data[0][0]
    fp = data[0][1]
    plt.plot(fp, tp, lw=lw, label=title)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False positive rate', fontsize=12)
plt.ylabel('True positive rate', fontsize=12)
#plt.title('Pi0/gamma classification')
plt.legend(loc="lower right", fontsize=12)
plt.xticks(ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=12)
plt.yticks(ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=12)

if args.output_name:
    plt.savefig(args.output_name + '.pdf')
    plt.savefig(args.output_name + '.png')
else:
    plt.show()
