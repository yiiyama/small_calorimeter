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

        x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, 1))

        x = tf.layers.conv2d(x, 25, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)

        nbinsx = self.nbinsx // 2
        nbinsy = self.nbinsy // 2

        x = tf.reshape(x, (self.batch_size, self.nbinsz, nbinsy, nbinsx, -1))

        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
        x = tf.reshape(x, (self.batch_size * nbinsy * nbinsx, -1))

        x = tf.layers.dense(x, units=self.nbinsz * 18, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.nbinsz * 10, activation=tf.nn.relu)

        x = tf.reshape(x, (self.batch_size, nbinsy, nbinsx, self.nbinsz, -1))
        x = tf.transpose(x, perm=[0, 3, 1, 2, 4])
        x = tf.reshape(x, (self.batch_size * self.nbinsz, nbinsy, nbinsx, -1))

        x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 5, 5, 6

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
