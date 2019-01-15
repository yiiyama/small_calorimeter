import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_2 import construct_sparse_io_dict, sparse_conv_collapse, sparse_conv_multi_neighbours, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions
from ops.sparse_conv import sparse_max_pool

MAXHITS = 2102
NUM_FEATURES = 9
SPATIAL_GLOBAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] # energy, layer(???) -> [0, 4]

class SparseConvMultiNeighborsModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_multi_neighbors', 'Multi neighbors')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_GLOBAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        feat = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.zeros([1], dtype=tf.int64))
        #feat = sparse_max_pool(feat, 70)
        feat = sparse_conv_collapse(feat)

        feat_list = []

        feat = self._batch_norm(feat)

        filters=42
        propagate=20
        dimensions=4 
        neighbours=40

        nsensors = [
            (1024, 40, 24),
            (512, 40, 24),
            (256, 40, 24),
            (128, 40, 24),
            (32, 40, 32),
            (1, 32, 32)
        ]
        
        #feat_list = []
        for nsen, nneigh, filt in nsensors:
            feat = sparse_conv_global_exchange(feat)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            feat = high_dim_dense(feat,64, activation=tf.nn.tanh)
            print(feat.shape)
            feat = sparse_conv_multi_neighbours(feat,
                                                n_neighbours=nneigh,
                                                n_dimensions=dimensions,
                                                n_filters=filt,
                                                n_propagate=propagate,
                                                total_distance=False)

            feat = self._batch_norm(feat)
            feat = max_pool_on_last_dimensions(feat, nsen)
            #feat_list.append(feat)

        #feat =  tf.concat(feat_list,axis=-1)
        #print('all feat',feat.shape) 
        #feat = tf.layers.dense(feat,128, activation=tf.nn.relu)
        #feat = tf.layers.dense(feat,3, activation=tf.nn.relu)
        #
        #features = sparse_conv_multi_neighbors(
        #    features,
        #    num_neighbors=16,
        #    #n_output=32,
        #    n_output=16,
        #    n_propagate=16,
        #    #n_filters=4*[20],
        #    n_filters=4*[10], 
        #    edge_filters=4*[2],
        #    space_transformations=[64,4],
        #    train_global_space=False)
        #
        #features = sparse_conv_global_exchange(features)
        #
        #features = self._batch_norm(features)
        #
        #features = tf.layers.dense(features, 32, activation=tf.nn.tanh)
        #
        #feat_list.append(features)
        #
        #features = sparse_conv_make_neighbors_simple_multipass(
        #    features,
        #    num_neighbors=16, 
        #    #n_output=32,
        #    n_output=16,
        #    n_propagate=16,
        #    #n_filters=4*[20],
        #    n_filters=4*[10], 
        #    edge_filters=4*[2],
        #    space_transformations=[32,4],
        #    train_global_space=False
        #)
        #
        ##features = sparse_conv_global_exchange(features)
        #
        #features = self._batch_norm(features)
        #feat_list.append(features)
        #
        #features = tf.concat(feat_list, axis=-1)
        print(feat.shape)
        feat = tf.reshape(feat, (self.batch_size, -1))
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, self.num_classes, activation=None)

        self.logits = feat
