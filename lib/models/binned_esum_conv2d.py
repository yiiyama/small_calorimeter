import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel

class BinnedESum2DConvModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'esum_2d_conv', '3D-binned energy sum 2D convolution')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, 1])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        with tf.variable_scope(self.variable_scope):
            x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, 1))

            x = tf.layers.conv2d(x, 25, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2)

            nfeatures = 18
            nbinsx = self.nbinsx // 2
            nbinsy = self.nbinsy // 2

            x = tf.reshape(x, (self.batch_size, self.nbinsz, nbinsy, nbinsx, nfeatures))

            x = tf.transpose(x, perm=[0, 2, 3, 4, 1])
            x = tf.reshape(x, (self.batch_size * nbinsy * nbinsx * nfeatures, self.nbinsz))

            nbinsz = self.nbinsz * 2
            
            x = tf.layers.dense(x, units=nbinsz, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=nbinsz, activation=tf.nn.relu)

            x = tf.reshape(x, (self.batch_size, nbinsy, nbinsx, nfeatures, nbinsz))
            x = tf.transpose(x, perm=[0, 4, 1, 2, 3])
            x = tf.reshape(x, (self.batch_size * nbinsz, nbinsy, nbinsx, nfeatures))

            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 5, 5, 6

            flattened_features = tf.reshape(x, (self.batch_size, -1))

            fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
            fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
            fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = fc_3
