import tensorflow as tf

from utils.noisy_eye import NoisyEyeInitializer
from utils.spatial import euclidean_squared, normalise_distance_matrix

def sparse_conv_seeded(input_features,
                       seed_indices,
                       seed_scaling,
                       nfilters,
                       nspacefilters=1, 
                       nspacedim=3,
                       nspacetransform=1,
                       add_to_orig=True,
                       seed_talk=True,
                       name=None,
                       returnmerged=False):
    '''
    first nspacetransform uses just the untransformed first <nspacedim> entries of the space coordinates
    '''
    if name is None:
        name="sparse_conv_seeded"

    all_features = tf.concat(input_features, axis=-1)
        
    trans_features = tf.layers.dense(all_features, nfilters, activation=tf.nn.relu)
    trans_features = tf.expand_dims(trans_features, axis=1)

    nbatch = all_features.shape[0]
    nvertex = all_features.shape[1]
    
    feature_layerout = []
    space_layerout = []

    nseeds = seed_indices.shape[1]
    batch = tf.range(nbatch, dtype=tf.int64)
    batch = tf.tile(batch[..., tf.newaxis, tf.newaxis], [1, nseeds, 1])
    seedselector = tf.concat((batch, seed_indices[..., tf.newaxis]), axis=-1)

    for _ in range(nspacetransform):
        trans_space = all_features 
        trans_space = tf.layers.dense(trans_space/10., nspacefilters, activation=tf.nn.tanh,
                                      kernel_initializer=NoisyEyeInitializer)
        trans_space = tf.layers.dense(trans_space, nspacedim, activation=tf.nn.tanh,
                                      kernel_initializer=NoisyEyeInitializer)
        trans_space = trans_space * 10

        space_layerout.append(trans_space)
        
        seed_trans_space_orig = tf.gather_nd(trans_space, seedselector)

        seed_trans_space = tf.expand_dims(seed_trans_space_orig, axis=2)
        seed_trans_space = tf.tile(seed_trans_space, [1, 1, nvertex, 1])

        all_trans_space = tf.expand_dims(trans_space, axis=1)
        #all_trans_space = tf.tile(all_trans_space, [1, seed_trans_space.shape[1], 1, 1])
        all_trans_space = tf.tile(all_trans_space, [1, nseeds, 1, 1]) # it's the same thing

        diff = all_trans_space - seed_trans_space
        diff = tf.reduce_sum(diff * diff, axis=-1)
        diff = normalise_distance_matrix(diff) 
        
        diff = tf.expand_dims(diff, axis=3)

        vertices_to_seed = diff * trans_features
        vertices_to_seed = tf.reduce_sum(vertices_to_seed, axis=2)

        #add back seed features
        seed_all_features = tf.gather_nd(all_features, seedselector)

        if seed_scaling is not None:
            seed_all_features = seed_scaling * seed_all_features
        
        #simple dense check this part
        #maybe add additional dense
        if seed_talk:
            #seed space transform?
            seed_distance = euclidean_squared(seed_trans_space_orig, seed_trans_space_orig)
            seed_distance = tf.expand_dims(seed_distance, axis=3)
            seed_update = seed_distance * tf.expand_dims(seed_all_features, axis=1)
            seed_update = tf.reduce_sum(seed_update, axis=2)
            seed_merged_features = tf.concat([seed_all_features, seed_update], axis=-1)
            seed_all_features = tf.layers.dense(seed_merged_features, seed_all_features.shape[2],
                                                activation=tf.nn.relu,
                                                kernel_initializer=NoisyEyeInitializer)
    
        seed_all_features = tf.concat([vertices_to_seed, seed_all_features], axis=-1)
        #propagate back

        # instead of computing something more, we simply take the full array of
        # seed features (all seeds concatenated) times distance as the message to the vertex        
        seed_to_vertices = tf.expand_dims(seed_all_features, axis=2)
        seed_to_vertices = seed_to_vertices * diff
        seed_to_vertices = tf.transpose(seed_to_vertices, perm=[0,2,1,3])
        seed_to_vertices = tf.reshape(seed_to_vertices, [seed_to_vertices.shape[0], seed_to_vertices.shape[1], -1])

        feature_layerout.append(seed_to_vertices)

    space_layerout = tf.concat(space_layerout, axis=-1)

    #combien old features with new ones
    feature_layerout = tf.concat(feature_layerout, axis=-1)
    feature_layerout = tf.concat([all_features, space_layerout, feature_layerout], axis=-1)
    feature_layerout = tf.layers.dense(feature_layerout/10., nfilters, activation=tf.nn.tanh, kernel_initializer=NoisyEyeInitializer)
    feature_layerout = feature_layerout * 10.

    if returnmerged:
        return feature_layerout
   
    return (feature_layerout, input_features[1], space_layerout)
