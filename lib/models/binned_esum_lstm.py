import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from models.classification import ClassificationModel

class BinnedESumLSTMModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'esum_lstm', '3D-binned E-sum LSTM')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, 1])]

    def _make_network(self):
        initializer = tf.random_uniform_initializer(-1, 1)

        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        lstm_cell = tf.nn.rnn_cell.LSTMCell(100, initializer=initializer)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(80, initializer=initializer)

        x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, 1))
        # [Nbatch * Nz, Ny, Nx, 1]

        x = tf.layers.conv2d(x, 25, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 18, [3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)

        nbinsx = self.nbinsx // 2
        nbinsy = self.nbinsy // 2

        x = tf.reshape(x, (self.batch_size, self.nbinsz, nbinsy, nbinsx, -1))
        # [Nbatch * Nz, Ny, Nx, 30]

        # consider each (x, y, ibatch) as independent data series
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
        x = tf.reshape(x, (self.batch_size * nbinsy * nbinsx, self.nbinsz, -1))
        # [Nbatch * Ny * Nx, Nz, 30]

        self.debug.append(('before_first_rnn', tf.shape(x)))

        x, state = tf.nn.dynamic_rnn(lstm_cell, x,
                                     dtype=tf.float32, scope="lstm_1")

        # [Nbatch * Ny * Nx, Nz, 100]

        self.debug.append(('after_first_rnn', tf.shape(x)))

        x, state = tf.nn.dynamic_rnn(lstm_cell_2, x,
                                     dtype=tf.float32, scope="lstm_2")

        # [Nbatch * Ny * Nx, Nz, 80]

        self.debug.append(('after_second_rnn', tf.shape(x)))

        x = tf.squeeze(tf.gather(x, [self.nbinsz - 1], axis=1), axis=1)

        # [Nbatch * Ny * Nx, 80]

        self.debug.append(('after_squeeze', tf.shape(x)))

        x = tf.reshape(x, (self.batch_size, nbinsy, nbinsx, -1))

        # [Nbatch, Ny, Nx, 80]

        x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 10, [3, 3], activation=tf.nn.relu, padding='same')

        x = tf.layers.max_pooling2d(x, [2, 2], strides=2) # 5, 5, 6

        x = tf.reshape(x, (self.batch_size, -1))

        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None)

        self.logits = x
