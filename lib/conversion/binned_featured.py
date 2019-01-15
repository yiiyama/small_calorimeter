from __future__ import print_function

import numpy as np
import tensorflow as tf

from conversion.base import Converter

class BinnedFeaturedConverter(Converter):
    def __init__(self):
        Converter.__init__(self)

        self.nbinsx = 16
        self.nbinsy = 16
        #self.nbinsz = 25
        self.nbinsz = 20
        self.nfeatures = 12

        self.xbins = np.array([-150. + i * 300. / self.nbinsx for i in range(self.nbinsx + 1)])
        self.ybins = np.array([-150. + i * 300. / self.nbinsy for i in range(self.nbinsy + 1)])
        self.zbins = np.array(range(self.nbinsz), dtype=np.float64)

        self.binned_data = np.zeros((self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures), dtype=np.float32)

    def init_convert(self):
        pass

    def convert(self, event):
        slot_used = np.zeros((self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures // 3), dtype=np.int32)

        self.binned_data.fill(0.)

        ixs = np.searchsorted(self.xbins, event[1], side='right')
        iys = np.searchsorted(self.ybins, event[2], side='right')
        izs = event[4].astype(np.int32)

        for ihit in range(len(event[0])):
            ix = ixs[ihit] - 1 # searchsorted returns the index to insert
            iy = iys[ihit] - 1
            iz = izs[ihit]

            dx = event[1][ihit] - (self.xbins[ix] + self.xbins[ix + 1]) * 0.5
            dy = event[2][ihit] - (self.ybins[iy] + self.ybins[iy + 1]) * 0.5

            i = 0
            while i != self.nfeatures // 3:
                if slot_used[iz][iy][ix][i] == 0:
                    break
                i += 1
            else:
                raise RuntimeError('All slots used: %d %d %d' % (ix, iy, iz))

            self.binned_data[iz][iy][ix][i * 3] = event[0][ihit]
            self.binned_data[iz][iy][ix][i * 3 + 1] = dx
            self.binned_data[iz][iy][ix][i * 3 + 2] = dy
            slot_used[iz][iy][ix][i] = 1

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
#        print(type(example['energy_map']))
#        print(np.shape(example['energy_map']))
#        print(example['energy_map'])
        energy_map = example['energy_map']
        esum = np.sum(energy_map[:,:,:,:,0:12:3], axis=4, keepdims=True)
        print(np.shape(esum))
        print(esum[0][0])
        print(type(example['labels_one_hot']))
        print(np.shape(example['labels_one_hot']))
        print(example['labels_one_hot'])
