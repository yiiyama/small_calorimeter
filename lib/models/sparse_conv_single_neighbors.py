import tensorflow as tf
import numpy as np
from models.sparse_conv_base import SparseConvModelBase
from ops.sparse_conv_2 import sparse_conv_collapse, sparse_conv_multi_neighbours, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions
from ops.sparse_conv import sparse_max_pool

class SparseConvSingleNeighborsModel(SparseConvModelBase):
    def __init__(self, config):
        SparseConvModelBase.__init__(self, config, 'sparse_conv_single_neighbors', 'Single neighbors')
        
    def _make_sparse_conv_network(self, feat):
        #feat = sparse_max_pool(feat, 70)
        feat = sparse_conv_collapse(feat)

        feat_list = []

        feat = self._batch_norm(feat)

        propagate=18
        dimensions=4 

        nsensors = [
            (1024, 40, 24),
            (512, 40, 24),
            (256, 40, 32),
            (128, 40, 32),
            (32, 40, 64),
            (1, 32, 64)
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
                                                total_distance=True,
                                                plus_mean=True)

            feat = self._batch_norm(feat)
            if nsen != feat.shape[1]:
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
