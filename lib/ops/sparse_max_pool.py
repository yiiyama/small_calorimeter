import tensorflow as tf

def sparse_max_pool(input_features, num_entries_result):
    other_features, spatial_features_global, spatial_features_local = input_features

    shape_spatial_features = spatial_features_global.get_shape().as_list()

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    n_batch = shape_spatial_features[0]

    # Neighbor matrix should be int as it should be used for indexing
    assert other_features.dtype in (tf.float64, tf.float32)

    _, I = tf.nn.top_k(tf.reduce_sum(other_features, axis=2), num_entries_result)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1, num_entries_result, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    out_other_features = tf.gather_nd(other_features, _indexing_tensor)
    out_spatial_features_global = tf.gather_nd(spatial_features_global, _indexing_tensor)
    out_spatial_features_local = tf.gather_nd(spatial_features_local, _indexing_tensor)

    return out_other_features, out_spatial_features_global, out_spatial_features_local
