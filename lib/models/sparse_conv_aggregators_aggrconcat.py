import tensorflow as tf
import numpy as np
from models.sparse_conv_base import SparseConvModelBase
from ops.sparse_conv_2 import sparse_conv_collapse, sparse_conv_hidden_aggregators, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions, zero_out_by_energy
from ops.sparse_conv import sparse_max_pool

class SparseConvAggregatorsAggrConcatModel(SparseConvModelBase):
    def __init__(self, config):
        SparseConvModelBase.__init__(self, config, 'sparse_conv_aggregators_aggrconcat', '')
        
    def _make_sparse_conv_network(self, feat):
        #feat = sparse_max_pool(feat, 70)
        feat = sparse_conv_collapse(feat)

        feat = zero_out_by_energy(feat)
        feat = self._batch_norm(feat)

        nlayers = 10
        
        aggregators = [10] * nlayers
        filters =     [12] * nlayers
        propagate =   [8] * nlayers
        pre_filters = [[]] * nlayers

        feat = sparse_conv_global_exchange(feat)
        feat = self._batch_norm(feat)
        feat = high_dim_dense(feat, 32, activation=tf.nn.tanh)

        aggr_list = []

        for i in range(nlayers):
            feat, aggr = sparse_conv_hidden_aggregators(feat, 
                                                  aggregators[i],
                                                  n_filters=filters[i],
                                                  pre_filters=pre_filters[i],
                                                  n_propagate=propagate[i],
                                                  plus_mean=True,
                                                  return_agg=True
                                                  )

            feat = self._batch_norm(feat)

            aggr_list.append(aggr)

        feat = tf.concat(aggr_list, axis=-1)

        feat = self._batch_norm(feat)

        print('concat feat_list', feat.shape)
        #feat = tf.transpose(feat, perm=[0, 2, 1])
        #feat = tf.reduce_sum(feat, axis=-1)
            
        print('reduced', feat.shape)
        feat = tf.reshape(feat, (self.batch_size, -1))
        feat = tf.layers.dense(feat, 16, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, self.num_classes, activation=None)

        self.logits = feat

