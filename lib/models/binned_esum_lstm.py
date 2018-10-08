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

        conv2d_arg_scope = slim.arg_scope([slim.conv2d],
                                          activation_fn=tf.nn.relu,
                                          weights_regularizer=slim.l2_regularizer(5.e-4),
                                          biases_initializer=tf.zeros_initializer(),
                                          padding='SAME')
        dense_common_args = {
            'kernel_initializer': tf.random_normal_initializer(mean=0., stddev=1.e-6),
            'bias_initializer': tf.random_normal_initializer(mean=0., stddev=1.e-6)
        }

        # [Nbatch, Nz, Ny, Nx, 1]
        x = self.placeholders[0]

        with tf.variable_scope(self.variable_scope):

            lstm_cell = tf.nn.rnn_cell.LSTMCell(100, initializer=initializer)
            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(80, initializer=initializer)

            with conv2d_arg_scope:
                x = tf.reshape(x, (self.batch_size * self.nbinsz, self.nbinsy, self.nbinsx, 1))
                # [Nbatch * Nz, Ny, Nx, 1]

                x = slim.conv2d(x, 30, [4, 4], scope='p2_c1')
                x = slim.conv2d(x, 30, [4, 4], scope='p2_c2')

                x = tf.reshape(x, (self.batch_size, self.nbinsz, self.nbinsy, self.nbinsx, -1))
                # [Nbatch * Nz, Ny, Nx, 30]

                # consider each (x, y, ibatch) as independent data series
                x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
                x = tf.reshape(x, (self.batch_size * self.nbinsy * self.nbinsx, self.nbinsz, -1))
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

                x = tf.reshape(x, (self.batch_size, self.nbinsy, self.nbinsx, -1))

                # [Nbatch, Ny, Nx, 80]

                x = slim.conv2d(x, 256, [1, 1], scope='e_mm', activation_fn=None)

            fc_0 = tf.reshape(x, (self.batch_size, -1))

            fc_1 = tf.layers.dense(fc_0, units=1024, activation=tf.nn.relu, **dense_common_args)
            fc_2 = tf.layers.dense(fc_1, units=1024, activation=tf.nn.relu, **dense_common_args)
            fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None, **dense_common_args)

            self.logits = fc_3
