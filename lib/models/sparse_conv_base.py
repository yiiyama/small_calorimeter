import tensorflow as tf
import numpy as np
import h5py
from models.classification import ClassificationModel
from ops.sparse_conv_2 import construct_sparse_io_dict

class SparseConvModelBase(ClassificationModel):
    MAXHITS = 2102
    NUM_FEATURES = 9
    SPATIAL_GLOBAL_FEATURES = [1, 2, 3] # x, y, z
    SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
    OTHER_FEATURES = [0] # energy, layer(???) -> [0, 4]
    
    def __init__(self, config, name, title):
        ClassificationModel.__init__(self, config, name, title)

        self._features = [
            ('rechit_data', tf.float32, [self.MAXHITS, self.NUM_FEATURES])
        ]

    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], self.OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], self.SPATIAL_GLOBAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], self.SPATIAL_LOCAL_FEATURES, axis=-1)
        
        feat_dict = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.zeros([1], dtype=tf.int64))

        self._make_sparse_conv_network(feat_dict)

    def _classification_add_evaluate_targets(self):
        other_features = tf.gather(self.placeholders[0], self.OTHER_FEATURES, axis=-1)
        sensor_energy = tf.reshape(other_features, (self.batch_size, -1))
        total_energy = tf.reduce_sum(sensor_energy, axis=-1) * 1.e-3
        
        self._evaluate_targets.append(total_energy)

        spatial_features_global = tf.gather(self.placeholders[0], self.SPATIAL_GLOBAL_FEATURES, axis=-1)
        sensor_x = spatial_features_global[:,:,0]
        sensor_x2 = sensor_x * sensor_x
        sensor_y = spatial_features_global[:,:,1]
        sensor_y2 = sensor_y * sensor_y
        sensor_r2 = sensor_x2 + sensor_y2
        sensor_r = tf.sqrt(sensor_r2)

        r_emean = tf.reduce_sum(sensor_energy * sensor_r, axis=-1) / tf.reduce_sum(sensor_energy, axis=-1)
        r2_emean = tf.reduce_sum(sensor_energy * sensor_r2, axis=-1) / tf.reduce_sum(sensor_energy, axis=-1)

        sigma_r = tf.sqrt(r2_emean - r_emean * r_emean)

        self._evaluate_targets.append(sigma_r)

        sensor_z = spatial_features_global[:,:,2]
        z_emean = tf.reduce_sum(sensor_energy * sensor_z, axis=-1) / tf.reduce_sum(sensor_energy, axis=-1)

        self._evaluate_targets.append(z_emean)

    def _classification_more_init_evaluate(self):
        self._energy_binning = np.arange(0., 110., 10., dtype=np.float32)
        self._energy_bin_all = np.zeros(np.shape(self._energy_binning)[0])
        self._energy_bin_correct = np.zeros(np.shape(self._energy_binning)[0])

        self._sigma_binning = np.arange(0., 60., 5., dtype=np.float32)
        self._sigma_bin_all = np.zeros(np.shape(self._sigma_binning)[0])
        self._sigma_bin_correct = np.zeros(np.shape(self._sigma_binning)[0])

        self._z_binning = np.arange(0., 200., 10., dtype=np.float32)
        self._z_bin_all = np.zeros(np.shape(self._z_binning)[0])
        self._z_bin_correct = np.zeros(np.shape(self._z_binning)[0])

        self._ntuples_file = h5py.File('%s/%s_ntuples.py' % (self.data_dir, self.variable_scope), 'w')
        self._ntuples = self._ntuples_file.create_dataset('ntuples', (0, 5), maxshape=(None, 5), chunks=(self.batch_size, 5))

    def _classification_more_do_evaluate(self, results, summary_dict):
        # limiting to num_classes 2
        truth, prob = results[:2]
        total_energy, sigma_r, z_emean = results[-3:]

        self._ntuples.resize(self._ntuples.shape[0] + self.batch_size, axis=0)

        row = []
        for elem in (truth.astype(np.float32), prob, total_energy, sigma_r, z_emean):
            row.append(np.reshape(elem, (self.batch_size, 1)))

        self._ntuples[-self.batch_size:] = np.concatenate(row, axis=1)

        energy_bins = np.searchsorted(self._energy_binning, total_energy)
        sigma_bins = np.searchsorted(self._sigma_binning, sigma_r)
        z_bins = np.searchsorted(self._z_binning, z_emean)

        for iex in range(self.batch_size):
            correct = (truth[iex] == 1 and prob[iex] >= 0.5) or (truth[iex] == 0 and prob[iex] < 0.5)

            b = energy_bins[iex]
            if b < 0 or b >= len(self._energy_binning):
                print('OOB energy', total_energy[iex])
                continue

            self._energy_bin_all[b] += 1
            if correct:
                self._energy_bin_correct[b] += 1

            b = sigma_bins[iex]
            if b < 0 or b >= len(self._sigma_binning):
                print('OOB sigma', sigma_r[iex])
                continue
                
            self._sigma_bin_all[b] += 1
            if correct:
                self._sigma_bin_correct[b] += 1

            b = z_bins[iex]
            if b < 0 or b >= len(self._z_binning):
                print('OOB z', z_emean[iex])
                continue
                
            self._z_bin_all[b] += 1
            if correct:
                self._z_bin_correct[b] += 1

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

        for ib, e in enumerate(self._sigma_binning):
            if self._sigma_bin_all[ib] == 0:
                continue
            
            print(e, self._sigma_bin_correct[ib] / self._sigma_bin_all[ib])

        data = [np.expand_dims(self._sigma_binning, 1)]
        data.append(np.expand_dims(self._sigma_bin_correct, 1))
        data.append(np.expand_dims(self._sigma_bin_all, 1))
        np.save('%s/%s_sigmadep.npy' % (self.data_dir, self.variable_scope), np.concatenate(data, axis=1))

        for ib, e in enumerate(self._z_binning):
            if self._z_bin_all[ib] == 0:
                continue
            
            print(e, self._z_bin_correct[ib] / self._z_bin_all[ib])

        data = [np.expand_dims(self._z_binning, 1)]
        data.append(np.expand_dims(self._z_bin_correct, 1))
        data.append(np.expand_dims(self._z_bin_all, 1))
        np.save('%s/%s_zdep.npy' % (self.data_dir, self.variable_scope), np.concatenate(data, axis=1))
