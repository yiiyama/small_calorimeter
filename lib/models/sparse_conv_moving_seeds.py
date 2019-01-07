import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv import sparse_max_pool
from ops.sparse_conv_2 import construct_sparse_io_dict, sparse_conv_collapse, sparse_conv_moving_seeds3

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0] # energy

class SparseConvMovingSeedsModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_moving_seeds', 'Sparse conv with moving seeds')

        #self.batch_norm_momentum = float(config['batch_norm_momentum'])
        #self.nfilters = int(config['nfilters']) #24
        #self.nspacefilters = int(config['nspacefilters']) #32
        #self.nspacedim = int(config['nspacedim']) #4
        #self.nrandom = int(config['nrandomseeds']) #1
        #self.nlayers = int(config['nconvlayers']) #11

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
        features = sparse_max_pool(features, 72)

        vertices = sparse_conv_collapse(features)

        n_filters = 64
        n_seed_dimensions = 4
        edge_multiplicity = 2

#        for n_seeds in [1, 2, 3, 3, 3, 3]:
        for n_seeds in [4]:
            vertices, _ = sparse_conv_moving_seeds3(vertices,
                                                    n_filters=n_filters,
                                                    n_seeds=n_seeds,
                                                    n_seed_dimensions=n_seed_dimensions,
                                                    seed_filters=[],
                                                    compress_before_propagate=True,
                                                    edge_multiplicity=edge_multiplicity)

            print('vertices', vertices.shape)

        x = tf.reshape(vertices, (self.batch_size, -1))
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
