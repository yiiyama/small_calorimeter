import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.generalized_conv import nearest_neighbor_conv, pooling_conv, pool_z

MAXHITS = 2679
NUM_FEATURES = 9

class UnbinnedGraphModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'unbinned_graph', 'Unbinned graph convolution')

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        # [Nbatch, Nhits, Nfeatures]
        x = self.placeholders[0]

        x = tf.gather(x, [0], axis=2) # energy only
        layer_conf = [4, 4, 8, 9]

        print('layer0')
        x = nearest_neighbor_conv(x, layer_conf, 32, 'layer0')

        print('layer0_x', x.shape)
        print('layer1')
        x = nearest_neighbor_conv(x, layer_conf, 32, 'layer1')
        print('layer1_x', x.shape)
        print('layer2')
        x = nearest_neighbor_conv(x, layer_conf, 16, 'layer2')
        print('layer2_x', x.shape)

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
        x = tf.reshape(x, (self.batch_size, layer_conf[3], -1))

        print('before_conv', x.shape)
        x = tf.layers.conv1d(x, 64, [2], strides=[2], padding='same')
        x = tf.layers.conv1d(x, 32, [2], strides=[2], padding='same')
        x = tf.layers.conv1d(x, 32, [2], strides=[2], padding='same')

        x = tf.reshape(x, (self.batch_size, -1))
        print('flattened_shape', x.shape)

        x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
