import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import construct_sparse_io_dict, sparse_max_pool
from ops.sparse_conv_bare_max import sparse_conv_bare_max
from ops.neighbors import indexing_tensor
from utils.params import get_num_parameters

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] #[0, 4] # energy, layer(???) -> [0, 4]

class SparseConvBareNoPoolModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_bare_nopool', 'Sparse conv with no space transformation, no pooling in between')

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

        # This model does not involve any spatial transform; compute the adjacency once and for all
        #static_indexing = indexing_tensor(spatial_features_global, num_neighbors)

        print('start', features['all_features'].shape)
        print(get_num_parameters(self.variable_scope))

        # Four consecutive conv. At each step, only all_features is updated
        
        features = sparse_conv_bare_max(features, neighbors=num_neighbors, output_all=48)
        layer0 = features['all_features']
        print('layer0', features['all_features'].shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_bare_max(features, neighbors=num_neighbors, output_all=48)
        layer1 = features['all_features']
        print('layer1', features['all_features'].shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_bare_max(features, neighbors=num_neighbors, output_all=48)
        layer2 = features['all_features']
        print('layer2', features['all_features'].shape)
        print(get_num_parameters(self.variable_scope))

        features = sparse_conv_bare_max(features, neighbors=num_neighbors, output_all=96)
        layer3 = features['all_features']
        print('layer3', features['all_features'].shape)
        print(get_num_parameters(self.variable_scope))

        # Final output is a concat of the outputs of the layers
        x = tf.concat([layer0, layer1, layer2, layer3], axis=-1)
        print('concat', x.shape)

        #pre_dense_shape = x.shape.as_list()
        #print('pre_dense', pre_dense_shape)
        #x = tf.reshape(x, [-1, pre_dense_shape[-1]])
        x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
        #x = tf.reshape(x, [pre_dense_shape[0], pre_dense_shape[1], 256])
        #print('post_dense', x.shape)
        print(get_num_parameters(self.variable_scope))
        x = tf.reduce_max(x, axis=1)

        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
