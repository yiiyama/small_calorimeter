import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_bare import sparse_conv_bare
from ops.sparse_max_pool import sparse_max_pool
from utils.spatial import make_indexing_tensor

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0, 4] # energy, layer(???) -> [0, 5]

class SparseConvBareModel(ClassificationModel):
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

        self.debug.append(('spatial', spatial_features_global))
        self.debug.append(('indexing', make_indexing_tensor(spatial_features_global, 3)))

        print('start', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer0')
        print('layer0', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer1')
        print('layer1', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer2')
        features = sparse_max_pool(features, 1000)
        print('layer2', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer3')
        features = sparse_max_pool(features, 500)
        print('layer3', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer4')
        features = sparse_max_pool(features, 250)
        print('layer4', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv_bare(features, num_neighbors=9, num_output_features=14, name='layer5')
        features = sparse_max_pool(features, 70)
        print('layer5', features[0].shape, features[1].shape, features[2].shape)

        x = tf.concat(features, axis=2)
        print('concat', x.shape)

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
