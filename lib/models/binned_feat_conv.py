import tensorflow as tf
import numpy as np
import h5py
from models.classification import ClassificationModel

class BinnedFeatured3DConvModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'featured_3d_conv', '3D-binned featured 3D convolution')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])
        self.nfeatures = int(config['nfeatures'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, Nfeat]
        x = self.placeholders[0]

        x = self._batch_norm(x)

        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [1, 1, 1], activation=tf.nn.relu, padding='same')

        x = tf.layers.conv3d(x, 36, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 18, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 18, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 12, 8, 8, 18

        x = self._batch_norm(x)

        x = tf.layers.conv3d(x, 20, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 6, 4, 4, 25

        x = self._batch_norm(x)

        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 3, 2, 2, 25

        x = self._batch_norm(x)

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=80, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x

        self.summary.append(('Logit 0', self.logits[0][0]))
        self.summary.append(('Logit 1', self.logits[0][1]))

    def _classification_add_evaluate_targets(self):
        hit_energies = tf.gather(self.placeholders[0], [0, 3, 6, 9], axis=-1)
        hit_energies = tf.reshape(hit_energies, (self.batch_size, -1))
        total_energy = tf.reduce_sum(hit_energies, axis=-1) * 1.e-3
        
        self._evaluate_targets.append(total_energy)

    def _classification_more_init_evaluate(self):
        self._energy_binning = np.arange(0., 110., 10., dtype=np.float32)
        self._energy_bin_all = np.zeros(np.shape(self._energy_binning)[0])
        self._energy_bin_correct = np.zeros(np.shape(self._energy_binning)[0])

        self._ntuples_file = h5py.File('%s/%s_ntuples.py' % (self.data_dir, self.variable_scope), 'w')
        self._ntuples = self._ntuples_file.create_dataset('ntuples', (0, 3), maxshape=(None, 3), chunks=(self.batch_size, 3))

    def _classification_more_do_evaluate(self, results, summary_dict):
        # limiting to num_classes 2
        truth, prob = results[:2]
        total_energy = results[-1]

        self._ntuples.resize(self._ntuples.shape[0] + self.batch_size, axis=0)

        row = []
        for elem in (truth.astype(np.float32), prob, total_energy):
            row.append(np.reshape(elem, (self.batch_size, 1)))

        self._ntuples[-self.batch_size:] = np.concatenate(row, axis=1)

        energy_bins = np.searchsorted(self._energy_binning, total_energy)

        for iex in range(self.batch_size):
            correct = (truth[iex] == 1 and prob[iex] >= 0.5) or (truth[iex] == 0 and prob[iex] < 0.5)

            b = energy_bins[iex]
            if b < 0 or b >= len(self._energy_binning):
                print('OOB energy', total_energy[iex])
                continue

            self._energy_bin_all[b] += 1
            if correct:
                self._energy_bin_correct[b] += 1

    def _classification_more_print_result(self):
        print('More results!')

        self._ntuples_file.close()

        data = []
        
        for ib, e in enumerate(self._energy_binning):
            if self._energy_bin_all[ib] == 0:
                continue
            
            print(e, self._energy_bin_correct[ib] / self._energy_bin_all[ib])

        data = [np.expand_dims(self._energy_binning, 1)]
        data.append(np.expand_dims(self._energy_bin_correct, 1))
        data.append(np.expand_dims(self._energy_bin_all, 1))
        np.save('%s/%s_energydep.npy' % (self.data_dir, self.variable_scope), np.concatenate(data, axis=1))
