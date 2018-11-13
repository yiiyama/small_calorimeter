import tensorflow as tf
import numpy as np
from models.classification import ClassificationModel
from ops.sparse_conv_seeded import sparse_conv_seeded
from ops.sparse_max_pool import sparse_max_pool

MAXHITS = 2679
NUM_FEATURES = 9
SPATIAL_FEATURES = [1, 2, 3] # x, y, z
SPATIAL_LOCAL_FEATURES = [6, 7] # vxy, vz
OTHER_FEATURES = [0, 5] # energy, layer(???) -> [0, 5]

class SparseConvSeededModel(ClassificationModel):
    def __init__(self, config):
        ClassificationModel.__init__(self, config, 'sparse_conv_seeded', 'Sparse conv with seed-driven adjacency')

        self.batch_norm_momentum = float(config['batch_norm_momentum'])
        self.nfilters = int(config['nfilters']) #24
        self.nspacefilters = int(config['nspacefilters']) #32
        self.nspacedim = int(config['nspacedim']) #4
        self.nrandom = int(config['nrandomseeds']) #1
        self.nlayers = int(config['nconvlayers']) #11

        self._features = [
            ('rechit_data', tf.float32, [MAXHITS, NUM_FEATURES])
        ]
        
    def _make_network(self):
        ph_other_features = tf.gather(self.placeholders[0], OTHER_FEATURES, axis=-1)
        ph_spatial_features_global = tf.gather(self.placeholders[0], SPATIAL_FEATURES, axis=-1)
        ph_spatial_features_local = tf.gather(self.placeholders[0], SPATIAL_LOCAL_FEATURES, axis=-1)

        features = (ph_other_features, ph_spatial_features_global, ph_spatial_features_local)

        seeds = tf.zeros([self.batch_size, 1], dtype=tf.int64)

        if self.nrandom > 0:
            random_seeds = tf.random_uniform(shape=(self.batch_size, self.nrandom), minval=0, maxval=MAXHITS, dtype=tf.int64)

            seeds = tf.concat([seeds, random_seeds], axis=-1)
            seeds = tf.transpose(seeds, [1,0])
            seeds = tf.random_shuffle(seeds)
            seeds = tf.transpose(seeds, [1,0])

        # with two random seeds
        #nfilters=24, space=32, spacedim=4, layers=11: batch 160, lr 0.002, approx 0.038 loss
        #nfilters=24, space=32, spacedim=6, layers=11: batch 140, lr 0.00013, 107530 paras, 
        #nfilters=24*1.5, space=32*1.5, spacedim=6, layers=5: batch 140, lr 0.00013, approx 100k paras, 
        # last two seem to not make a big difference.. but latter seems slightly slower in converging
        # but with more potential maybe?
        # deeper one 0.04 at 26k, 0.045 at 26k
        # same config (just 6 layers) without random seeds: 
        
        features = (
            tf.layers.batch_normalization(features[0], momentum=self.batch_norm_momentum),
            tf.layers.batch_normalization(features[1], momentum=self.batch_norm_momentum),
            tf.layers.batch_normalization(features[2], momentum=self.batch_norm_momentum)
        )

        for i in range(self.nlayers):
            features = sparse_conv_seeded(features,
                                          seeds,
                                          None,
                                          nfilters=self.nfilters,
                                          nspacefilters=self.nspacefilters,
                                          nspacetransform=1,
                                          nspacedim=self.nspacedim)

            features = (
                tf.layers.batch_normalization(features[0], momentum=self.batch_norm_momentum),
                tf.layers.batch_normalization(features[1], momentum=self.batch_norm_momentum),
                tf.layers.batch_normalization(features[2], momentum=self.batch_norm_momentum)
            )

        features = sparse_max_pool(features, 60)

        x = tf.concat(features, axis=2)

        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=30, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=self.num_classes, activation=None) # (Batch, Classes)

        self.logits = x
