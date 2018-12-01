import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_2 import construct_sparse_io_dict, sparse_conv_make_neighbors2, max_pool_on_last_dimensions
from ops.activations import sinc, gauss_times_linear

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0, 4] # energy, layer(???) -> [0, 4]
ALL_FEATURES = [1, 2, 3, 0, 4, 6, 7]

class SparseConvNeighborMPIModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_neighbor_mpi', 'Sparse conv with message passing within neighbors')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        features = tf.gather(self.placeholders[0], ALL_FEATURES, axis=-1)
        print('input', features.shape)

        log_energy = True

        space_global = features[:,:,:3]
        colours = features[:,:,3:5]
        space_local = features[:,:,5:]

        space_global=tf.concat([tf.expand_dims(space_global[:,:,0]/150.,axis=2),
                                       tf.expand_dims(space_global[:,:,1]/150.,axis=2),
                                       tf.expand_dims(space_global[:,:,2]/150.,axis=2)],
                                      axis=-1)

        if log_energy:
            colours = tf.log(colours + 1.) / 10.
        else:
            colours = colours * 1.e-4
    
        space_local = space_local/150.

        features = tf.concat([space_global, colours, space_local], axis=-1)

        features = sparse_conv_make_neighbors2(features, num_neighbors=16, output_all=[16] * 8, space_transformations=3, train_space=False)
        print('layer0', features.shape)

        features = max_pool_on_last_dimensions(features, 4, 500)
        print('pooled', features.shape)

        features = sparse_conv_make_neighbors2(features, num_neighbors=32, output_all=[32] * 16, space_transformations=3, train_space=False)
        print('layer1', features.shape)

        features = max_pool_on_last_dimensions(features, 4, 200)

        features = sparse_conv_make_neighbors2(features, num_neighbors=64, output_all=[16] * 12, space_transformations=3, train_space=False)
        print('layer1', features.shape)

        features = max_pool_on_last_dimensions(features, 4, 50)

        x = tf.reshape(features, (self.batch_size, -1))

        #x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        #x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
