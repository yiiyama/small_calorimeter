#!/usr/bin/env python

try:
    import setGPU
except ImportError:
    pass

import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Merge converted TFRecords')
parser.add_argument('converter', help="Converter class")
parser.add_argument('output', help="Output file name")
parser.add_argument('input', nargs = '+', help="Path to input tfrecords")
args = parser.parse_args()

sys.argv = []

BATCH_SIZE = 1000
NUM_CLASSES = 2

import numpy as np
import tensorflow as tf

import conversion

converter = getattr(conversion, args.converter)()

feature_mapping = converter.feature_mapping()
feature_mapping['labels_one_hot'] = tf.FixedLenFeature((NUM_CLASSES,), tf.int64)

def parse_one(example_proto):
    parsed = tf.parse_single_example(example_proto, feature_mapping)
    return parsed

dataset = tf.data.TFRecordDataset(args.input, compression_type = 'GZIP') \
                 .map(parse_one) \
                 .batch(BATCH_SIZE)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

writer = tf.python_io.TFRecordWriter(args.output,
                                     options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

with tf.Session() as sess:
    iter = 0
    while True:
        try:
            mapped, = sess.run([next_element])
        except tf.errors.OutOfRangeError:
            break

        for ib in range(min(BATCH_SIZE, np.shape(mapped['labels_one_hot'])[0])):
            feature = {}
            for key, value in mapped.iteritems():
                if value.dtype == np.float32:
                    data = tf.train.FloatList(value = value[ib].flatten())
                    feature[key] = tf.train.Feature(float_list = data)
                elif value.dtype == np.int64:
                    data = tf.train.Int64List(value = value[ib].flatten())
                    feature[key] = tf.train.Feature(int64_list = data)
                else:
                    raise RuntimeError('Dtype of ' + key + ' not float32 or int64')
        
            example = tf.train.Example(features = tf.train.Features(feature = feature))
        
            writer.write(example.SerializeToString())

writer.close()
