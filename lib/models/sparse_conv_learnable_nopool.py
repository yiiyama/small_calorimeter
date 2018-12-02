import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import sparse_conv_make_neighbors2, sparse_max_pool, construct_sparse_io_dict

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] # energy

class SparseConvLearnableNeighborsNoPoolModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_learnable_nopool', 'Sparse conv with learnable adjacency')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        # Normalize the input
        other_features = other_features * 1.e-4
        spatial_features_global = spatial_features_global / 150. # Z also normalized by 150 to emphasize the first layers
        spatial_features_local = spatial_features_local / 150.

        # Truncate the input down to first 512 most energetic hits
        features = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.zeros([1], dtype=tf.int64))
        features = sparse_max_pool(features, 512)

        # Then redefine all_features to be really all features
        features['all_features'] = tf.concat([features['all_features'], features['spatial_features_global'], features['spatial_features_local']], axis=-1)

        num_neighbors = 15

        features = sparse_conv_make_neighbors2(
            features,
            num_neighbors=num_neighbors,
            output_all=32,
            space_transformations=[10, 10, 10],
            propagrate_ahead=True,
            name='layer0'
        )
        layer0 = features['all_features']

        features = sparse_conv_make_neighbors2(
            features,
            num_neighbors=num_neighbors,
            output_all=32,
            space_transformations=[10, 10, 10],
            propagrate_ahead=True,
            name='layer1'
        )
        layer1 = features['all_features']

        features = sparse_conv_make_neighbors2(
            features,
            num_neighbors=num_neighbors,
            output_all=32,
            space_transformations=[10, 10, 10],
            propagrate_ahead=True,
            name='layer2'
        )
        layer2 = features['all_features']

        features = sparse_conv_make_neighbors2(
            features,
            num_neighbors=num_neighbors,
            output_all=32,
            space_transformations=[3],
            propagrate_ahead=False,
            name='layer3'
        )
        layer3 = features['all_features']

        x = tf.concat([layer0, layer1, layer2, layer3], axis=-1)
        print('concat', x.shape)

        x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
        x = tf.reduce_max(x, axis=1)

#        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
