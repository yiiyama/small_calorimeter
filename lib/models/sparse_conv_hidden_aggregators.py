import tensorflow as tf
import numpy as np
from models.sparse_conv_base import SparseConvModelBase
from ops.sparse_conv_2 import sparse_conv_collapse, sparse_conv_hidden_aggregators, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions, zero_out_by_energy
from ops.sparse_conv import sparse_max_pool

class SparseConvHiddenAggregatorsModel(SparseConvModelBase):
    def __init__(self, config):
        SparseConvModelBase.__init__(self, config, 'sparse_conv_hidden_aggregators', '')
        
    def _make_sparse_conv_network(self, feat):
        #feat = sparse_max_pool(feat, 70)
        feat = sparse_conv_collapse(feat)

        feat = zero_out_by_energy(feat)
        feat = self._batch_norm(feat)
        
        aggregators = 7*[4]  
        filters =     7*[40]
        nsensors =    [1024] + [512] + (3*[64]) + [32, 1]
        propagate =   7*[32]
        pre_filters = 7*[[]]
        
        feat = sparse_conv_global_exchange(feat)
        feat = self._batch_norm(feat)
        feat = high_dim_dense(feat,32, activation=tf.nn.tanh)
        feat_list=[]
        for i in range(len(filters)):
            feat = sparse_conv_hidden_aggregators(feat, 
                                                  aggregators[i],
                                                  n_filters=filters[i],
                                                  pre_filters=pre_filters[i],
                                                  n_propagate=propagate[i],
                                                  plus_mean=True
                                                  )
            feat = self._batch_norm(feat)
            if nsensors[i] != feat.shape[1]:
                feat = max_pool_on_last_dimensions(feat, nsensors[i])
            
        print(feat.shape)
        feat = tf.reshape(feat, (self.batch_size, -1))
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, self.num_classes, activation=None)

        self.logits = feat
