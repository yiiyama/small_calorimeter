import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from utils.graph_conv import nearest_neighbor_conv, pooling_conv

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

        with tf.variable_scope(self.variable_scope):
            print('layer0')
            x = nearest_neighbor_conv(x, layer_conf, 20, 'layer0')
            print('layer1')
            x = nearest_neighbor_conv(x, layer_conf, 10, 'layer1')
            print('layer2')
            x = nearest_neighbor_conv(x, layer_conf, 10, 'layer2')

            print('layer3')
            x, layer_conf = pooling_conv(x, layer_conf, 10, 'layer3')

            # reduce the size of representation by reductive convolution rather than pooling
            #x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
#
#            nfeatures = 18
#            nbinsx = self.nbinsx // 2
#            nbinsy = self.nbinsy // 2
#
#            x = tf.reshape(x, (self.batch_size, self.nbinsz, nbinsy, nbinsx, nfeatures))
#
#            x = tf.transpose(x, perm=[0, 2, 3, 4, 1])
#            x = tf.reshape(x, (self.batch_size * nbinsy * nbinsx * nfeatures, self.nbinsz))
#
#            nbinsz = self.nbinsz * 2
#            
#            x = tf.layers.dense(x, units=nbinsz, activation=tf.nn.relu)
#            x = tf.layers.dense(x, units=nbinsz, activation=tf.nn.relu)
#
#            x = tf.reshape(x, (self.batch_size, nbinsy, nbinsx, nfeatures, nbinsz))
#            x = tf.transpose(x, perm=[0, 4, 1, 2, 3])
#            x = tf.reshape(x, (self.batch_size * nbinsz, nbinsy, nbinsx, nfeatures))
#
#            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
#            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
#
#            x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 5, 5, 6
#
            flattened_features = tf.reshape(x, (self.batch_size, -1))

            fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
            fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
            fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = fc_3
