from __future__ import print_function

import numpy as np
import tensorflow as tf

class Converter(object):
    def __init__(self):
        pass

    def init_convert(self):
        pass

    def convert(self, event):
        return {}

    def feature_mapping(self):
        return {}

    def print_example(self, example):
        print(example)

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

class BinnedFeaturedConverter(Converter):
    def __init__(self):
        Converter.__init__(self)

        self.nbinsx = 16
        self.nbinsy = 16
        self.nbinsz = 25
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
#        print(type(example['labels_one_hot']))
#        print(np.shape(example['labels_one_hot']))
        print(example['labels_one_hot'])
        energy_map = example['energy_map']
        esum = np.sum(energy_map[:,:,:,:,0:12:3], axis=4, keepdims=True)
        print(np.shape(esum))
        print(esum[0][0])
