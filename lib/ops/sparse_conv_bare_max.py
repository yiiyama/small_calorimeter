import tensorflow as tf
from ops.neighbors import indexing_tensor
from ops.nn import *
from ops.sparse_conv import construct_sparse_io_dict

def sparse_conv_bare_max(sparse_dict, neighbors=10, output_all=15):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :return: Dictionary containing output which can be made input to the next layer
    """

    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    if type(neighbors) is int:
        _indexing_tensor = indexing_tensor(spatial_features_global, neighbors)
    else:
        _indexing_tensor = neighbors

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    pre_output = tf.layers.dense(gathered_all, output_all, activation=tf.nn.relu)
    output = tf.reduce_max(pre_output, axis=-2)

    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)
