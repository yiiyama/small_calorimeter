import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel

class BinnedFeatured2DConvModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'featured_2d_conv', '3D-binned featured 2D convolution')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])
        self.nfeatures = int(config['nfeatures'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        with tf.variable_scope(self.variable_scope):
            x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures))

            x = tf.layers.conv2d(x, 50, [1, 1], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 25, [1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.conv2d(x, 25, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2)

            print('after first pooling', x.shape)

            nbinsx = self.nbinsx // 2
            nbinsy = self.nbinsy // 2

            x = tf.reshape(x, (self.batch_size, self.nbinsz, nbinsy, nbinsx, -1))

            print('reshaped', x.shape)

            x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
            x = tf.reshape(x, (self.batch_size * nbinsy * nbinsx, -1))

            print('reshaped for dense', x.shape)

            x = tf.layers.dense(x, units=self.nbinsz, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=self.nbinsz // 2, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=self.nbinsz // 4, activation=tf.nn.relu)

            print('after second dense', x.shape)

            x = tf.reshape(x, (self.batch_size, nbinsy, nbinsx, self.nbinsz // 4, -1))
            x = tf.transpose(x, perm=[0, 3, 1, 2, 4])
            x = tf.reshape(x, (self.batch_size * (self.nbinsz // 4), nbinsy, nbinsx, -1))

            print('reshaped', x.shape)

            x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 5, 5, 6

            print('after second pooling', x.shape)

            x = tf.reshape(x, (self.batch_size, -1))

            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = x
