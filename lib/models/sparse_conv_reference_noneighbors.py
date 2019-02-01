import tensorflow as tf
import numpy as np
from models.sparse_conv_base import SparseConvModelBase
from ops.sparse_conv_2 import sparse_conv_collapse, sparse_conv_multi_neighbours, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions, zero_out_by_energy
from ops.sparse_conv import sparse_max_pool

class SparseConvReferenceNoNeighborsModel(SparseConvModelBase):
    def __init__(self, config):
        SparseConvModelBase.__init__(self, config, 'sparse_conv_reference_noneighbors', 'Single neighbors')
        
    def _make_sparse_conv_network(self, feat):
        #feat = sparse_max_pool(feat, 70)
        feat = sparse_conv_collapse(feat)
        feat = zero_out_by_energy(feat)

        feat_list = []

        feat = self._batch_norm(feat)

        feat = tf.layers.dense(feat, 64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 64, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 2, activation=tf.nn.relu)

        print(feat.shape)
        feat = tf.reshape(feat, (self.batch_size, -1))
        feat = tf.layers.dense(feat, 12, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, self.num_classes, activation=None)

        self.logits = feat
