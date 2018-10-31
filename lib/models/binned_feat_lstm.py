import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from models.classification import ClassificationModel

class BinnedFeaturedLSTMModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'featured_lstm', '3D-binned featured LSTM')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])
        self.nfeatures = int(config['nfeatures'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures])]

    def _make_network(self):
        initializer = tf.random_uniform_initializer(-1, 1)

        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        with tf.variable_scope(self.variable_scope):
            x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures))

            x = tf.layers.conv2d(x, 50, [1, 1], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 25, [1, 1], activation=tf.nn.relu, padding='same')

            x = tf.layers.conv2d(x, 25, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 12, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 25, 8, 8, 12

            nbinsx = self.nbinsx // 2
            nbinsy = self.nbinsy // 2

            x = tf.layers.conv2d(x, 12, [3, 3], activation=tf.nn.relu, padding='same')
            x = tf.layers.conv2d(x, 8, [3, 3], activation=tf.nn.relu, padding='same')

            x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 25, 4, 4, 12

            x = tf.reshape(x, (self.batch_size, self.nbinsz, -1))

            lstm_cell = tf.nn.rnn_cell.LSTMCell(80, initializer=initializer)
            x, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, scope="lstm_1")

            print('after first RNN', x.shape)

            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(40, initializer=initializer)
            x, state = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype=tf.float32, scope="lstm_2")

            print('after second RNN', x.shape)

            x = tf.squeeze(tf.gather(x, [self.nbinsz - 1], axis=1), axis=1) # 50

            print('after squeeze', x.shape)

            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

            self.logits = x
