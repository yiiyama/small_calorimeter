from __future__ import print_function

import numpy as np
import tensorflow as tf

from conversion.base import Converter

class UnbinnedPaddingConverter(Converter):
    # Pad hit list with zeros (for <= prod5 datasets)

    MAXHITS = 2679
    NUM_FEATURES = 9

    def convert(self, event):
        nhits = len(event[0])
        zeros = np.zeros(UnbinnedConverter.MAXHITS - nhits, dtype = np.float32)

        rechits = np.concatenate([np.expand_dims(np.append(event[i], zeros), axis = 1) for i in xrange(UnbinnedConverter.NUM_FEATURES)], axis = 1)

        rechit_data = tf.train.FloatList(value = rechits.flatten())
        num_entries = tf.train.Int64List(value = np.array([nhits], dtype = np.int64))
    
        return {
            'rechit_data': tf.train.Feature(float_list = rechit_data),
            'num_entries': tf.train.Feature(int64_list = num_entries)
        }
