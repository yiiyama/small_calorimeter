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
        x = self.placeholders

        x = sparse_conv(x, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True)
        x = sparse_conv(x, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True)
        x = sparse_conv(x, num_neighbors=18, output_features=15, space_transformations=[10, 10, 10], propagate_ahead=True)
        x = sparse_conv(x, num_neighbors=18, output_features=30, space_transformations=[10, 10, 10], propagate_ahead=True)
        x = sparse_conv(x, num_neighbors=18, output_features=3, space_transformations=[3], propagate_ahead=True)

        # Need to cook this down to classification!

        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
