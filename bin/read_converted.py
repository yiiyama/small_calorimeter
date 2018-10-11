#!/usr/bin/env python

try:
    import setGPU
except ImportError:
    pass

import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Read converted records')
parser.add_argument('converter', help="Converter class")
parser.add_argument('input', help="Path to input tfrecords")
parser.add_argument('nevents', nargs = '?', default = 1, type = int, help = 'Number of events to read')
args = parser.parse_args()

sys.argv = []

import numpy as np
import tensorflow as tf

import conversion

converter = getattr(conversion, args.converter)()

feature_mapping = converter.feature_mapping()
feature_mapping['labels_one_hot'] = tf.FixedLenFeature((2,), tf.int64)
#feature_mapping['labels_one_hot'] = tf.FixedLenFeature((6,), tf.int64)
#feature_mapping['labels_one_hot'] = tf.VarLenFeature(tf.int64)

def parse_one(example_proto):
    parsed = tf.parse_single_example(example_proto, feature_mapping)
    return parsed

dataset = tf.data.TFRecordDataset([args.input], compression_type = 'GZIP') \
                 .map(parse_one) \
                 .batch(args.nevents)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    example, = sess.run([next_element])

    converter.print_example(example)
