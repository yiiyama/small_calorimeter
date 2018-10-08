import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel

class BinnedESum3DConvModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'esum_3d_conv', '3D-binned energy sum 3D convolution')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, 1])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        with tf.variable_scope(self.variable_scope):
            x = tf.layers.conv3d(x, 25, [1, 3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.conv3d(x, 18, [1, 3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.conv3d(x, 18, [1, 3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 8, 8, 13, 18

            x = tf.layers.conv3d(x, 18, [1, 3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 4, 4, 7, 18

            x = tf.layers.conv3d(x, 18, [1, 3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 2, 2, 4, 18

            x = tf.layers.conv3d(x, 18, [1, 2, 2], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv3d(x, 18, [2, 1, 1], activation=tf.nn.relu, padding='same')

            flattened_features = tf.reshape(x, (self.batch_size, -1))

            fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
            fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
            fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = fc_3
