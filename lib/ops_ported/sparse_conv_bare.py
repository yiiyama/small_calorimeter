import tensorflow as tf
from utils.spatial import make_indexing_tensor

def sparse_conv_bare(input_features, num_neighbors=10, num_output_features=15, name=None):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :return: Dictionary containing output which can be made input to the next layer
    """

    if name is None:
        name = "sparse_conv_bare"

    other_features, spatial_features_global, spatial_features_local = input_features

    shape_other_features = other_features.get_shape().as_list()
    shape_space_features = spatial_features_global.get_shape().as_list()
    shape_space_features_local = spatial_features_local.get_shape().as_list()

    # All of these tensors should be 3-dimensional
    assert len(shape_space_features) == 3 and len(shape_other_features) == 3 and len(shape_space_features_local) == 3

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_other_features[0]
    assert shape_space_features[1] == shape_other_features[1]
    assert shape_space_features[0] == shape_space_features_local[0]
    assert shape_space_features[1] == shape_space_features_local[1]

    indexing_tensor, _ = make_indexing_tensor(spatial_features_global, num_neighbors)
    shape_indexing_tensor = indexing_tensor.get_shape().as_list()

    assert len(shape_indexing_tensor) == 4

    # Neighbor matrix should be int as it should be used for indexing
    assert indexing_tensor.dtype == tf.int64

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]

    gathered_other = tf.gather_nd(other_features, indexing_tensor)  # [B,E,5,F]

    pre_output = tf.layers.dense(gathered_other, num_output_features, activation=tf.nn.relu)
    output = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), num_output_features, activation=tf.nn.relu)

    return output, spatial_features_global, spatial_features_local
