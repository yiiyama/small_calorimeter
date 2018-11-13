import tensorflow as tf

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    assert (A.dtype in (tf.float32, tf.float64)) and (B.dtype in (tf.float32, tf.float64))

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()
    
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]
    
    #just exploit broadcasting
    B_trans = tf.transpose(tf.expand_dims(B,axis=3), perm=[0, 3, 1, 2])
    A_exp = tf.expand_dims(A,axis=2)
    diff = A_exp-B_trans
    distance = tf.reduce_sum(tf.square(diff),axis=-1)
    #to avoid rounding problems and keep it strict positive
    distance= tf.abs(distance)

    return distance


def normalise_distance_matrix(AdMat):
    maxAdMat = tf.reduce_max(tf.reduce_max(AdMat, axis=-1, keepdims=True), axis=-1, keepdims=True)
    AdMat = AdMat / maxAdMat
    AdMat = (tf.zeros_like(AdMat) + 1) - AdMat
    scaling = tf.reduce_sum(tf.reduce_mean(AdMat, axis=-1, keepdims=False))
    AdMat = AdMat / scaling 

    return AdMat


def make_nearest_neighbor_matrix(spatial_features, k=10):
    """
    Nearest neighbors matrix given spatial features.

    :param spatial_features: Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch, S = spatial features
    :param k: Max neighbors
    :return:
    """

    # Neighbor matrix should be int as it should be used for indexing
    assert spatial_features.dtype in (tf.float64, tf.float32)

    D = euclidean_squared(spatial_features, spatial_features)
    values, indices = tf.nn.top_k(-D, k)

    return indices, values


def make_indexing_tensor(spatial_features, k=10, n_batch=-1):
    """
    Return the indexing tensor and the distance matrix.
    Indexing tensor is used in gather_nd:
    Idx[b][s][n] = (b, nth neighbor of s)
    gather_nd(features, Idx)[b][s][n] = (features of nth neighbor of s in batch b)
    note: 0th neighbor is itself
    """

    # Neighbor matrix should be int as it should be used for indexing
    assert spatial_features.dtype in (tf.float64, tf.float32)

    shape = spatial_features.get_shape().as_list()

    # All of these tensors should be 3-dimensional
    assert len(shape) == 3

    n_max_entries = shape[1]

    neighbor_matrix, distance_matrix = make_nearest_neighbor_matrix(spatial_features, k) # [B, N, k]
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3) # [B, N, k, 1]
    
    if n_batch > 0:
        expanded_neighbor_matrix = tf.tile(expanded_neighbor_matrix, [n_batch, 1, 1, 1])
    else:
        n_batch = shape[0]

    batch_range = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1, n_max_entries, k, 1])

    indexing_tensor = tf.concat([batch_range, expanded_neighbor_matrix], axis=3)

    return tf.cast(indexing_tensor, tf.int64), distance_matrix
