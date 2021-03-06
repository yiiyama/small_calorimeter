import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import sparse_conv
from ops.sparse_max_pool import sparse_max_pool

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0, 5] # energy, layer(???) -> [0, 5]

class SparseConvLearnableNeighborsModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_learnable', 'Sparse conv with learnable adjacency')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_FEATURES, axis=-1)
        spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        features = (other_features, spatial_features_global, spatial_features_local)

        print('start', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, num_output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer0')
        print('layer0', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, num_output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer1')
        print('layer1', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, num_output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer2')
        print('layer2', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, num_output_features=30, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer3')
        print('layer3', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, num_output_features=15, space_transformations=[3], propagate_ahead=False, name='layer4')
        print('layer4', features[0].shape, features[1].shape, features[2].shape)

        features = sparse_max_pool(features, 200)
        print('maxpool', features[0].shape, features[1].shape, features[2].shape)

        x = tf.concat(features, axis=2)
        print('concat', x.shape)

        x = tf.reshape(x, (self.batch_size, -1))

#        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
#        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
