import tensorflow as tf

from utils.noisy_eye import NoisyEyeInitializer
from utils.spatial import make_indexing_tensor

def sparse_conv(input_features,
                num_neighbors=10, 
                output_features=15,
                space_transformations=[10, 10, 10],
                propagate_ahead=False,
                strict_global_space=True,
                name = None):
    """
    Defines sparse convolutional layer
    
    --> revise the space distance stuff

    :param input_features: (other, spatial, spatial_local) features
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_features: Number of output features for color like outputs. If list, create multiple dense layers with output of hidden layer i = output_features[i]
    :return: new (other, spatial, spatial_local) features
    """
    if name is None:
        name="sparse_conv"

    assert type(space_transformations) is list

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

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]

    ## Spatial transformation: concatenate global & local features and run them through several dense layers

    transformed_space_features = tf.concat([spatial_features_global, spatial_features_local], axis=-1)

    if strict_global_space:
        # Spatial features are static within the batch - use the first entry to reduce memory footprint
        transformed_space_features = transformed_space_features[0,:,:]
        transformed_space_features = tf.expand_dims(transformed_space_features,axis=0)

    for ist, st in enumerate(space_transformations):
        transformed_space_features = tf.layers.dense(transformed_space_features,
                                                     st, 
                                                     activation=(tf.nn.tanh if ist == len(space_transformations) - 1 else None),
                                                     kernel_initializer=NoisyEyeInitializer,
                                                     name='%s_sp_%d' % (name, ist))
            
    ## Find nearest neighbors (Euclidean) of each node in the transformed space

    if strict_global_space:
        # need to tell the number of batches to make_indexing_tensor
        create_indexing_batch = n_batch
    else:
        # make_indexing_tensor can figure out the number of batches
        create_indexing_batch = -1

    # see doc of make_indexing_tensor for the meaning of indexing tensor
    indexing_tensor, distance_matrix = make_indexing_tensor(transformed_space_features,
                                                            num_neighbors,
                                                            create_indexing_batch)

    if strict_global_space:
        # Revert batch reduction
        transformed_space_features = tf.tile(transformed_space_features, [n_batch, 1, 1])
    
    # distance_matrix is strict negative
    # -distance_matrix -> strict positive, small = near
    # softsign(-distance_matrix) -> (-1, 1), small = near
    # 1 - softsign(-distance_matrix) -> (0, 1), large = near
    inverse_distance = 1. - tf.nn.softsign(-distance_matrix) # *float(num_neighbors)
    inverse_distance = tf.expand_dims(inverse_distance, axis=3)

    ## Flatten all "other features" (i.e. energy etc.) of nearest neighbors for each sensor and
    ## pass them through dense layers

    if type(output_features) is int:
        output_features = [output_features]
   
    for ift, nft in enumerate(output_features):
        gathered = tf.gather_nd(other_features, indexing_tensor) * inverse_distance
        flattened_gathered = tf.reshape(gathered, [n_batch, n_max_entries, -1])
        other_features = tf.layers.dense(flattened_gathered, nft, activation=tf.nn.relu, name='%s_%d_%d' % (name, nft, ift))

    if propagate_ahead:
        spatial_features_global = tf.concat([transformed_space_features, spatial_features_global], axis=-1)
      
    return other_features, spatial_features_global, spatial_features_local
