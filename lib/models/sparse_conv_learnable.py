import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import sparse_conv

MAXHITS = 2679
NUM_SPATIAL = 3
NUM_SPATIAL_LOCAL = 2
NUM_OTHER = 2

class SparseConvLearnableNeighborsModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_learnable', 'Sparse conv with learnable adjacency')

        self._features = [
            ('features_other', tf.float32, [MAXHITS, NUM_OTHER]),
            ('features_spatial', tf.float32, [MAXHITS, NUM_SPATIAL]),
            ('features_spatial_local', tf.float32, [MAXHITS, NUM_SPATIAL_LOCAL])
        ]
        
    def _make_network(self):
        features = self.placeholders[:3]

        print('start', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer0')
        print('layer0', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer1')
        print('layer1', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer2')
        print('layer2', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, output_features=30, space_transformations=[10, 10, 10], propagate_ahead=True, name='layer3')
        print('layer3', features[0].shape, features[1].shape, features[2].shape)
        features = sparse_conv(features, num_neighbors=18, output_features=3, space_transformations=[3], propagate_ahead=True, name='layer4')

        #x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)
        x = tf.reshape(features[0], (self.batch_size, -1))
        x = tf.gather(x, [0, 1], axis = 1)

        self.logits = x
