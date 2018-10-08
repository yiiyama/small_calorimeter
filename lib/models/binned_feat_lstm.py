import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from models.classification import ClassificationModel

class BinnedFeaturedLSTMModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'feat_lstm', '3D-binned featured LSTM')

        self.nbinsx = int(config['nbinsx'])
        self.nbinsy = int(config['nbinsy'])
        self.nbinsz = int(config['nbinsz'])
        self.nfeatures = int(config['nfeatures'])

        self._features = [('energy_map', tf.float32, [self.nbinsz, self.nbinsy, self.nbinsx, self.nfeatures])]

    def _make_network(self):
        # [Nbatch, Nz, Ny, Nx, Nfeat]
        x = self.placeholders[0]

        initializer = tf.random_uniform_initializer(-1, 1)

        conv2d_arg_scope = slim.arg_scope([slim.conv2d],
                                          activation_fn=tf.nn.relu,
                                          weights_regularizer=slim.l2_regularizer(weight_decay),
                                          biases_initializer=tf.zeros_initializer(),
                                          padding='SAME')
        dense_arg_scope = slim.arg_scope([tf.layers.dense],
                                         kernel_initializer=tf.random_normal_initializer(mean=0., stddev=1.e-6),
                                         bias_initializer=tf.random_normal_initializer(mean=0., stddev=1.e-6))

        with tf.variable_scope(self.variable_scope):

            lstm_cell = tf.nn.rnn_cell.LSTMCell(100, initializer=initializer)
            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(80, initializer=initializer)

            with conv2d_arg_scope:
                x = slim.conv2d(x, self.nfeatures * 2, [1, 1], scope='p1_c1')
                x = slim.conv2d(x, self.nfeatures * 2, [1, 1], scope='p1_c2')

                x = slim.conv2d(x, 30, [5, 5], scope='p2_c1')
                x = slim.conv2d(x, 30, [5, 5], scope='p2_c2')

                x = tf.reshape(x, (self.batch_size, self.nbinsz, -1))

                x, state = tf.nn.dynamic_rnn(lstm_cell, x,
                                             dtype=tf.float32, scope="lstm_1")

                self.debug.append(('after_first_rnn_x', x))
                self.debug.append(('after_first_rnn_state', state))

                x, state = tf.nn.dynamic_rnn(lstm_cell_2, x,
                                             dtype=tf.float32, scope="lstm_2")

                self.debug.append(('after_second_rnn_x', x))
                self.debug.append(('after_second_rnn_state', state))

                x = tf.squeeze(tf.gather(x, [self.nbinsz - 1], axis=1))

                x = slim.conv2d(x, 256, [1, 1], scope='e_mm', activation_fn=None)

            with dense_arg_scope:
                fc_0 = tf.reshape(x, (self.batch_size, -1))

                fc_1 = tf.layers.dense(fc_0, units=1024, activation=tf.nn.relu)
                fc_2 = tf.layers.dense(fc_1, units=1024, activation=tf.nn.relu)
                fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

            self.logits = fc_3
