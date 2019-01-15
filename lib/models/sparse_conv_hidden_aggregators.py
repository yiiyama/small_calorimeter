import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_2 import construct_sparse_io_dict, sparse_conv_collapse, sparse_conv_hidden_aggregators, sparse_conv_global_exchange, high_dim_dense, max_pool_on_last_dimensions, zero_out_by_energy
from ops.sparse_conv import sparse_max_pool

MAXHITS = 2102
NUM_FEATURES = 9
SPATIAL_GLOBAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] # energy, layer(???) -> [0, 4]

class SparseConvHiddenAggregatorsModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_hidden_aggregators', '')

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
            feat = max_pool_on_last_dimensions(feat, nsensors[i])
            
        print(feat.shape)
        feat = tf.reshape(feat, (self.batch_size, -1))
        feat = tf.layers.dense(feat, 32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, self.num_classes, activation=None)

        self.logits = feat
