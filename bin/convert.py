#!/usr/bin/env python

import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Convert input ROOT tree to TFRecords')
parser.add_argument('converter', help="Converter class")
parser.add_argument('input', help="Path to input tree")
parser.add_argument('output', help="Output file name")
parser.add_argument('nevents', nargs = '?', default = -1, type = int, help = 'Number of events to process')
parser.add_argument('--epi-filter', '-F', action = 'store_true', dest = 'epi_filter', help = 'Select events with electron or charged pion only.')
args = parser.parse_args()

sys.argv = []

import ROOT
ROOT.gROOT.SetBatch(True)
import root_numpy as rnp
import numpy as np
import tensorflow as tf

import conversion

converter = getattr(conversion, args.converter)()
converter.init_convert()

source = ROOT.TFile.Open(args.input)
tree = source.Get('B4')

feature_branches = [
    'rechit_energy',
    'rechit_x',
    'rechit_y',
    'rechit_z',
    'rechit_layer',
    'rechit_varea',
    'rechit_vxy',
    'rechit_vz'
]
if tree.GetBranch('rechit_detid'):
    feature_branches.append('rechit_detid')

num_features = len(feature_branches)

label_branches = [
    'isElectron',
    'isPionNeutral'
]
if not args.epi_filter:
    label_branches += [
        'isMuon',
        'isPionCharged',
        'isK0Long',
        'isK0Short'
    ]
num_labels = len(label_branches)

if args.epi_filter:
    selection = 'isElectron || isPionNeutral'
else:
    selection = None

full_array = rnp.tree2array(tree, feature_branches + label_branches, selection)

writer = tf.python_io.TFRecordWriter(args.output,
                                     options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

print 'Start conversion'

ievent = 0
for event in full_array:
    if ievent == args.nevents:
        break

    ievent += 1

    if ievent % 100 == 1:
        print ievent, 'events'

    labels = np.concatenate([np.expand_dims(event[i].astype(np.int64), axis = 0) for i in xrange(num_features, num_features + num_labels)], axis = 0)

    feature = converter.convert(event)

    labels_one_hot = tf.train.Int64List(value = labels.flatten())
    feature['labels_one_hot'] = tf.train.Feature(int64_list = labels_one_hot)

    example = tf.train.Example(features = tf.train.Features(feature = feature))

    writer.write(example.SerializeToString())

writer.close()
