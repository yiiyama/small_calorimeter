import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_2 import construct_sparse_io_dict, sparse_conv_collapse, sparse_conv_make_neighbors_simple_multipass, sparse_conv_global_exchange
from ops.sparse_conv import sparse_max_pool

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] # energy, layer(???) -> [0, 4]

class SparseConvNeighborSimpleMultipassModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_neighbors_simple_multipass', 'Make neighbors simple multipass')

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

        features = construct_sparse_io_dict(other_features, spatial_features_global, spatial_features_local, tf.zeros([1], dtype=tf.int64))
        features = sparse_max_pool(features, 70)
        features = sparse_conv_collapse(features)

        feat_list = []

        features = self._batch_norm(features)

        features = sparse_conv_make_neighbors_simple_multipass(
            features,
            num_neighbors=16,
            #n_output=32,
            n_output=16,
            n_propagate=16,
            #n_filters=4*[20],
            n_filters=4*[10], 
            edge_filters=4*[2],
            space_transformations=[64,4],
            train_global_space=False)
        
        features = sparse_conv_global_exchange(features)

        features = self._batch_norm(features)

        features = tf.layers.dense(features, 32, activation=tf.nn.tanh)

        feat_list.append(features)

        features = sparse_conv_make_neighbors_simple_multipass(
            features,
            num_neighbors=16, 
            #n_output=32,
            n_output=16,
            n_propagate=16,
            #n_filters=4*[20],
            n_filters=4*[10], 
            edge_filters=4*[2],
            space_transformations=[32,4],
            train_global_space=False
        )

        #features = sparse_conv_global_exchange(features)
        
        features = self._batch_norm(features)
        feat_list.append(features)
        
        features = tf.concat(feat_list, axis=-1)
        print(features.shape)
        features = tf.reshape(features, (self.batch_size, -1))
        features = tf.layers.dense(features, 20, activation=tf.nn.relu)
        features = tf.layers.dense(features, self.num_classes, activation=None)

        self.logits = features
