from __future__ import print_function

import numpy as np
import tensorflow as tf

from conversion.base import Converter

class BinnedESumConverter(Converter):
    def __init__(self):
        Converter.__init__(self)

        self.nbinsx = 16
        self.nbinsy = 16
        self.nbinsz = 25

        self.binned_data = np.zeros((self.nbinsz, self.nbinsy, self.nbinsx, 1), dtype=np.float32)

    def init_convert(self):
        import ROOT

        xbinning = (self.nbinsx, -150., 150.)
        ybinning = (self.nbinsy, -150., 150.)
        zbinning = (self.nbinsz, 0., 25.)

        self.histogram = ROOT.TH3D('esum', '', *(xbinning + ybinning + zbinning))

    def convert(self, event):
        self.histogram.Reset()

        for ihit in range(len(event[0])):
            self.histogram.Fill(event[1][ihit], event[2][ihit], event[4][ihit], event[0][ihit])

        for iz in range(self.histogram.GetNbinsZ()):
            for iy in range(self.histogram.GetNbinsY()):
                for ix in range(self.histogram.GetNbinsX()):
                    self.binned_data[iz][iy][ix][0] = self.histogram.GetBinContent(ix + 1, iy + 1, iz + 1)

        energy_map = tf.train.FloatList(value = self.binned_data.flatten())

        return {
            'energy_map': tf.train.Feature(float_list = energy_map)
        }

    def feature_mapping(self):
        feature_shape = np.shape(self.binned_data)

        return {
            'energy_map': tf.FixedLenFeature(feature_shape, tf.float32)
        }

    def print_example(self, example):
        print(example['energy_map'][0][0])
#        print(type(example['energy_map']))
#        print(np.shape(example['energy_map']))
#        print(example['energy_map'])
#        print(type(example['labels_one_hot']))
#        print(np.shape(example['labels_one_hot']))
#        print(example['labels_one_hot'])
