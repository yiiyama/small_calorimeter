import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel

class UnbinnedSpacialTransformModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'featured_3d_conv', '3D-binned featured 3D convolution')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])
        self.nfeatures = int(config['nfeatures'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, Nfeat]
        x = self.placeholders[0]

        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [1, 1, 1], activation=tf.nn.relu, padding='same')

        x = tf.layers.conv3d(x, 18, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 18, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 18, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 12, 8, 8, 18

        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 6, 4, 4, 25

        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv3d(x, 25, [2, 3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 3, 2, 2, 25

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
