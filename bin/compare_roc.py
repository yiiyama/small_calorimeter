import os
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser(description='Compare ROC curves')
parser.add_argument('input', nargs = '+', help="Paths to npy files")
parser.add_argument('--output', '-o', dest = 'output_name', help='Output file name (no extension)')

args = parser.parse_args()

plt.figure()

lw = 2

for fname in args.input:
    with open(fname, 'rb') as source:
        data = np.load(source)

    tp = data[0][0]
    fp = data[0][1]
    plt.plot(fp, tp, lw=lw, label=os.path.basename(fname).replace('.npy', ''))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Electron identification')
plt.legend(loc="lower right")

if args.output_name:
    plt.savefig(args.output_name + '.pdf')
    plt.savefig(args.output_name + '.png')
else:
    plt.show()
