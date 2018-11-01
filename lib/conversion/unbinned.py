from __future__ import print_function

import numpy as np
import tensorflow as tf

from conversion.base import Converter

class UnbinnedConverter(Converter):
    MAXHITS = 2679
    NUM_FEATURES = 9

    def convert(self, event):
        rechits = np.concatenate([np.expand_dims(event[i].astype(np.float32), axis = 1) for i in xrange(UnbinnedConverter.NUM_FEATURES)], axis = 1)

        rechit_data = tf.train.FloatList(value = rechits.flatten())
    
        return {
            'rechit_data': tf.train.Feature(float_list = rechit_data)
        }

    def feature_mapping(self):
        return {
            'rechit_data': tf.FixedLenFeature((UnbinnedConverter.MAXHITS, UnbinnedConverter.NUM_FEATURES), tf.float32)
        }

    def print_example(self, example):
        print(type(example['rechit_data']))
        print(np.shape(example['rechit_data']))
        print(example['rechit_data'])
        print(type(example['labels_one_hot']))
        print(np.shape(example['labels_one_hot']))
        print(example['labels_one_hot'])
