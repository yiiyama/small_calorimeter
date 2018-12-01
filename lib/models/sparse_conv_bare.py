import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import construct_sparse_io_dict, sparse_conv_bare, sparse_max_pool

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0, 4] # energy, layer(???) -> [0, 4]

class SparseConvBareModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_bare', 'Sparse conv with no space transformation')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        num_neighbors = 15

        self.debug('spatial', spatial_features_global)

        features = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.constant(MAXHITS, dtype=tf.int64))

        print('start', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        print('layer0', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        print('layer1', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        features = sparse_max_pool(features, 1000)
        print('layer2', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        features = sparse_max_pool(features, 500)
        print('layer3', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        features = sparse_max_pool(features, 250)
        print('layer4', features['all_features'].shape)

        features = sparse_conv_bare(features, num_neighbors=num_neighbors, output_all=14)
        features = sparse_max_pool(features, 70)
        print('layer5', features['all_features'].shape)

        x = tf.concat([features['all_features'], features['spatial_features_global'], features['spatial_features_local']], axis=2)
        print('concat', x.shape)

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
