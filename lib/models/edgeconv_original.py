import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import construct_sparse_io_dict, sparse_max_pool
from ops.sparse_conv_test import sparse_conv_globalrel
from utils.params import get_num_parameters

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] #[0, 4] # energy, layer(???) -> [0, 4]

class EdgeConvOriginalModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'edge_conv_original', 'Original implementation of EdgeConv')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        other_features = other_features * 1.e-4
        spatial_features_global = spatial_features_global / 150.
        spatial_features_local = spatial_features_local / 150.

        features = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.constant(MAXHITS, dtype=tf.int64))

        features = sparse_max_pool(features, 512)

        features['all_features'] = tf.concat([features['all_features'], features['spatial_features_local']], axis=-1)

        num_neighbors = 15

        print('start', features['all_features'].shape)

        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=48, concat_all=True)
        layer0 = features['spatial_features_global']
        print('layer0', layer0.shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=48)
        layer1 = features['spatial_features_global']
        print('layer1', layer1.shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=48)
        layer2 = features['spatial_features_global']
        print('layer2', layer2.shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=96)
        layer3 = features['spatial_features_global']
        print('layer3', layer3.shape)
        print(get_num_parameters(self.variable_scope))
        
#        features = sparse_max_pool(features, 1000)
#        print('layer2', features['spatial_features_global'].shape)
#
#        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=64)
#        features = sparse_max_pool(features, 500)
#        print('layer3', features['spatial_features_global'].shape)
#
#        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=64)
#        features = sparse_max_pool(features, 250)
#        print('layer4', features['spatial_features_global'].shape)
#
#        features = sparse_conv_globalrel(features, num_neighbors=num_neighbors, n_output=64)
#        features = sparse_max_pool(features, 64)
#        print('layer5', features['spatial_features_global'].shape)

        x = tf.concat([layer0, layer1, layer2, layer3], axis=-1)
        pre_dense_shape = x.shape.as_list()
        print('pre_dense', pre_dense_shape)
        x = tf.reshape(x, [-1, pre_dense_shape[-1]])
        x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
        x = tf.reshape(x, [pre_dense_shape[0], pre_dense_shape[1], 256])
        print('post_dense', x.shape)
        x = tf.reduce_max(x, axis=1)
        print(get_num_parameters(self.variable_scope))

        #x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
