import tensorflow as tf
import numpy as np
from models.binary_classification import BinaryClassificationModel
import tensorflow.contrib.slim as slim

class BinningModel(BinaryClassificationModel):
    def __init__(self, config):
        BinaryClassificationModel.__init__(self, config, 'h3d_conv_1', 'Dummy 3d conv net')

        self.dim_x = int(config['input_dim_x'])
        self.dim_y = int(config['input_dim_y'])
        self.dim_z = int(config['input_dim_z'])
        self.num_input_features = int(config['input_features'])

    def _make_feature_inputmap(self):
        return [('x', tf.FixedLenFeature((self.dim_x, self.dim_y, self.dim_z, self.num_input_features), tf.float32))]

    def _make_feature_placeholders(self):
        return [
            tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.dim_x, self.dim_y, self.dim_z, self.num_input_features])
        ]

    def _make_network(self):
        # [Batch, Height, Width, Depth, Channels]
        x = self.placeholders[0]

        weight_decay=0.0005

        with tf.variable_scope(self.variable_scope()):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                    x = tf.layers.conv3d(0.001 * x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
                    x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
                    x = tf.layers.conv3d(x, 50, [1, 1, 1], activation=tf.nn.leaky_relu, padding='same')
    
                    x = tf.layers.conv3d(x, 25, [3, 3, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')
    
                    x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')
    
                    x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')
    
                    x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 10, 10, 12
    
                    x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')
    
                    x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 5, 5, 6
    
                    x = tf.layers.conv3d(x, 18, [3, 3, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 5], activation=tf.nn.relu, padding='same')
    
                    x = tf.layers.max_pooling3d(x, [2, 2, 2], strides=2) # 2, 2, 3
    
                    x = tf.layers.conv3d(x, 18, [2, 2, 1], activation=tf.nn.relu, padding='same')
                    x = tf.layers.conv3d(x, 18, [1, 1, 3], activation=tf.nn.relu, padding='same')
    
                    flattened_features = tf.reshape(x, (self.batch_size, -1))
    
                    fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
                    fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
                    fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None) # (Batch, Classes)
    
                    return fc_3

