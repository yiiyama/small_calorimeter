import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from utils.graph_conv_weightwise import layer_sizes, nearest_neighbor_conv, pooling_conv, pool_z

MAXHITS = 2679
NUM_FEATURES = 9

class UnbinnedGraphZDenseModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'unbinned_graph_zdense', 'Unbinned graph convolution with Z dense')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        # [Nbatch, Nhits, Nfeatures]
        x = self.placeholders[0]

        x = tf.gather(x, [0], axis=2) # energy only
        layer_conf = [4, 4, 8, 9]
        nlayers = sum(layer_conf)

        with tf.variable_scope(self.variable_scope):
            print('layer0')
            x = nearest_neighbor_conv(x, layer_conf, 25, 'layer0')

            print('layer0_x', x.shape)
            print('layer1')
            x = nearest_neighbor_conv(x, layer_conf, 12, 'layer1')
            print('layer1_x', x.shape)

            print('layer3')
            # reduce the size of representation by reductive convolution rather than pooling
            x, layer_conf = pooling_conv(x, layer_conf, 16, 'layer3')
            print('layer3_x', x.shape)
            print('layer3_conf', layer_conf)

            print('layer4')
            x, layer_conf = pooling_conv(x, layer_conf, 16, 'layer4')
            print('layer4_x', x.shape)
            print('layer4_conf', layer_conf)

            # all layers now have the same geometry
            x = tf.reshape(x, (self.batch_size, nlayers, layer_sizes[3], 16))
            print('reshape', x.shape)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            print('transpose', x.shape)
            x = tf.reshape(x, (self.batch_size * layer_sizes[3], -1))
            print('reshape', x.shape)

            x = tf.layers.dense(x, nlayers * 5, activation=tf.nn.relu)
            print('dense', x.shape)
            nlayers = nlayers // 2
            x = tf.layers.dense(x, nlayers * 5, activation=tf.nn.relu)
            print('dense', x.shape)
            nlayers = nlayers // 2
            x = tf.layers.dense(x, nlayers * 5, activation=tf.nn.relu)
            print('dense', x.shape)
            nlayers = nlayers // 2
            x = tf.layers.dense(x, nlayers * 5, activation=tf.nn.relu)
            print('dense', x.shape)

            x = tf.reshape(x, (self.batch_size, layer_sizes[3], nlayers, 5))
            print('reshape', x.shape)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            print('transpose', x.shape)
            x = tf.reshape(x, (self.batch_size, nlayers * layer_sizes[3], 5))
            print('reshape', x.shape)

            x = nearest_neighbor_conv(x, [0, 0, 0, nlayers], 10, 'layer5')
            x = nearest_neighbor_conv(x, [0, 0, 0, nlayers], 10, 'layer6')

            x = nearest_neighbor_conv(x, [0, 0, 0, nlayers], 15, 'layer7')
            x = nearest_neighbor_conv(x, [0, 0, 0, nlayers], 4, 'layer8')

#            x = tf.reshape(x, (self.batch_size, nlayers, layer_sizes[3], 10))
#            x = tf.reshape(x, (self.batch_size, nlayers, layer_sizes[3] * 10))
#
#            x = tf.layers.dense(x, 25, activation=tf.nn.relu)
#            x = tf.layers.dense(x, 25, activation=tf.nn.relu)

            x = tf.reshape(x, (self.batch_size, -1))
            print('flattened_shape', x.shape)

            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = x
