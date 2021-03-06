"""
Generalized 2D convolution where sensor cells are considered as vertices connected to other vertices
by named edges. Edges carry convolution weights.
This implementation: [nsens, nch_in] -> [nsens, nch_out] as a single matrix operation. Input is reformed
into [nsens + 1, nch_in] where input[nsens][i] = 0 for all i. Input is then gathered by the sensor id for each output sensor for each edge type. Since the input and output number of sensors are identical, after a reshape we get [nsens, nch_in * nweight] where the value is 0 for non-existing sensor-weight combination (achieved by a trick of gathering nsens'th input by default). The weights is a trainable variable of shape [nch_in * nweight, nch_out], so the matmul of the two results in the desired output.
"""

import tensorflow as tf
import numpy as np

NSENSORS = 2679

layer_sizes = [304, 171, 76, 19]
bases = [4, 3, 2, 1]

class AddWeight(object):
    def __init__(self, name, wnames, nch_out, nsens_out, nch_in, nsens_in):
        # using identical graph edges at each layers - should check if there will be a difference
        # if we use different weights for different layer types

        self.nch_in = nch_in

        self.weights = tf.get_variable(name, shape=[len(wnames) * nch_in, nch_out], trainable=True)

        self.weight_map = dict((wn, i) for i, wn in enumerate(wnames))
        self.gather_indices = np.full([nsens_out, len(wnames)], nsens_in, dtype=np.int32)

        self.out_index = -1

    def add(self, idx, wname):
        wid = self.weight_map[wname]
        self.gather_indices[self.out_index][wid] = idx


def nearest_neighbor_conv_kernel(input, layer_conf, nout, name):
    """
    Create a NN graph convolution kernel as a matrix with shape [Nsens_in x nout, Nsens_in x nchannels]
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    wnames = ['Lself', 'LleftL', 'LdownL', 'LrightL', 'LupL', 'Sself', 'SleftS', 'SdownS', 'SrightS', 'SupS', 'LrightS0', 'LrightS1',  'LrightS2', 'LrightS3', 'LupS0', 'LupS1', 'LupS2', 'LupS3', 'SleftL', 'SdownL']

    adder = AddWeight(name, wnames, nout, input.shape[1], input.shape[2], input.shape[1])

    layer = 0
    for itype in range(4):
        base = bases[itype]

        nlargecol = base * 2
        nleft = nlargecol * base
        nlarge = nleft + base * base
        nsmallcol = base * 4
        nsmall = nsmallcol * nsmallcol
        ntotal = nlarge + nsmall

        while layer - sum(layer_conf[:itype]) < layer_conf[itype]:
            for i in range(nlarge):
                adder.out_index = i
                adder.add(i, 'Lself')
    
                if i < nlargecol:
                    pass
                elif i < nleft + base:
                    adder.add(i - nlargecol, 'LleftL')
                else:
                    adder.add(i - base, 'LleftL')
            
                if (i < nleft and i % nlargecol != 0) or (i >= nleft and i % base != 0):
                    adder.add(i - 1, 'LdownL')
        
                if i < nleft - base:
                    adder.add(i + nlargecol, 'LrightL')
                elif i >= nleft and i < nlarge - base:
                    adder.add(i + base, 'LrightL')
        
                if (i < nleft and i % nlargecol != (nlargecol - 1)) or (i >= nleft and i % base != (base - 1)):
                    adder.add(i + 1, 'LupL')
        
            for i in range(nleft - base, nleft):
                adder.out_index = i
                offset = nlarge + (i - (nleft - base)) * 4
            
                adder.add(offset, 'LrightS0')
                adder.add(offset + 1, 'LrightS1')
                adder.add(offset + 2, 'LrightS2')
                adder.add(offset + 3, 'LrightS3')
        
            for i in range(nleft + base, nlarge, base):
                adder.out_index = i
                offset = nlarge + (i - (nleft + base)) * 16
            
                adder.add(offset, 'LupS0')
                adder.add(offset + base * 4, 'LupS1')
                adder.add(offset + base * 8, 'LupS2')
                adder.add(offset + base * 12, 'LupS3')
        
            for i in range(nlarge, ntotal):
                adder.out_index = i
                adder.add(i, 'Sself')
            
                if (i - nlarge) % nsmallcol != 0:
                    adder.add(i - 1, 'SdownS')
        
                if i >= nlarge + nsmallcol:
                    adder.add(i - nsmallcol, 'SleftS')
            
                if (i - nlarge) % nsmallcol != (nsmallcol - 1):
                    adder.add(i + 1, 'SupS')
            
                if i < ntotal - nsmallcol:
                    adder.add(i + nsmallcol, 'SrightS')
        
            for i in range(nlarge, nlarge + nsmallcol):
                adder.out_index = i
                adder.add((i - nlarge) // 4 + nleft - base, 'SleftL')
            
            for i in range(nlarge, ntotal, nsmallcol):
                adder.out_index = i
                adder.add(((i - nlarge) // (4 * nsmallcol)) * base + nleft + base, 'SdownL')

            layer += 1

    return adder.weights, adder.gather_indices

def nearest_neighbor_conv(input, layer_conf, nout, name):
    """
    Perform a NN convolution on input.
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    batch_size = input.shape[0]
    nsens = input.shape[1]
    nch_in = input.shape[2]

    weights, gather_indices = nearest_neighbor_conv_kernel(input, layer_conf, nout, name)
    bias = tf.get_variable('%s_bias' % name, shape = [nout], trainable = True, initializer = tf.zeros_initializer())

    x = tf.concat([input, tf.constant(np.zeros([batch_size, 1, nch_in], dtype=np.float32))], axis=1)
    x = tf.gather(x, np.reshape(gather_indices, [-1]), axis=1)
    x = tf.reshape(x, [batch_size, nsens, -1])
    x = tf.tensordot(x, weights, [[2], [0]])
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.relu(x)

    return x


def pooling_conv_kernel(input, layer_conf, nout, name):
    """
    Create a pooling graph convolution kernel as a matrix with shape [Nsens_in x nchannels, Nsens_out, nout]
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    wnames = ['Llb', 'Llt', 'Lrt', 'Lrb', 'Slb', 'Slt', 'Srt', 'Srb', 'Lself', 'Lleft', 'Ldown', 'Lright', 'Lup', 'Sself', 'Sleft', 'Sdown', 'Sright', 'Sup']

    nsens_in = 0
    nsens_out = 0
    out_layer_conf = [0, 0, 0, 0]
    for itype in range(4):
        # geometry of the output layer
        if itype == 0:
            outtype = 2
        else:
            outtype = 3

        nsens_in += layer_sizes[itype] * layer_conf[itype]
        nsens_out += layer_sizes[outtype] * layer_conf[itype]

        out_layer_conf[outtype] += layer_conf[itype]

    adder = AddWeight(name, wnames, nout, nsens_out, input.shape[2], nsens_in)

    layer = 0
    for itype in range(4):
        # geometry of the output layer
        if itype == 0:
            base = 2
        else:
            base = 1

        nlargecol = base * 2
        nleft = nlargecol * base
        nlarge = nleft + base * base
        nsmallcol = base * 4
        nsmall = nsmallcol * nsmallcol
        ntotal = nlarge + nsmall

        # iterate over input layers
        while layer - sum(layer_conf[:itype]) < layer_conf[itype]:
            if itype == 0 or itype == 2:
                for i in range(nleft):
                    adder.out_index = i

                    outcol = i // nlargecol
                    incol0 = outcol * 2
                    incol1 = outcol * 2 + 1
                    outrow = i % nlargecol
                    inrow0 = outrow * 2
                    inrow1 = outrow * 2 + 1
                    nincol = nlargecol * 2

                    adder.add(incol0 * nincol + inrow0, 'Llb')
                    adder.add(incol0 * nincol + inrow1, 'Llt')
                    adder.add(incol1 * nincol + inrow0, 'Lrb')
                    adder.add(incol1 * nincol + inrow1, 'Lrt')

                for i in range(nleft, nlarge):
                    adder.out_index = i

                    outcol = (i - nleft) // base
                    incol0 = outcol * 2
                    incol1 = outcol * 2 + 1
                    outrow = (i - nleft) % base
                    inrow0 = outrow * 2
                    inrow1 = outrow * 2 + 1
                    ninleft = nleft * 4
                    nincol = base * 2

                    adder.add(incol0 * nincol + inrow0 + ninleft, 'Llb')
                    adder.add(incol0 * nincol + inrow1 + ninleft, 'Llt')
                    adder.add(incol1 * nincol + inrow0 + ninleft, 'Lrb')
                    adder.add(incol1 * nincol + inrow1 + ninleft, 'Lrt')
            
                for i in range(nlarge, ntotal):
                    adder.out_index = i

                    outcol = (i - nlarge) // nsmallcol
                    incol0 = outcol * 2
                    incol1 = outcol * 2 + 1
                    outrow = (i - nlarge) % nsmallcol
                    inrow0 = outrow * 2
                    inrow1 = outrow * 2 + 1
                    ninlarge = nlarge * 4
                    nincol = nsmallcol * 2

                    adder.add(incol0 * nincol + inrow0 + ninlarge, 'Slb')
                    adder.add(incol0 * nincol + inrow1 + ninlarge, 'Slt')
                    adder.add(incol1 * nincol + inrow0 + ninlarge, 'Srb')
                    adder.add(incol1 * nincol + inrow1 + ninlarge, 'Srt')

            elif itype == 1:
                for i in range(nleft):
                    adder.out_index = i

                    outcol = i // nlargecol
                    incol0 = outcol * 3
                    incol1 = outcol * 3 + 1
                    incol2 = outcol * 3 + 2
                    outrow = i % nlargecol
                    inrow0 = outrow * 3
                    inrow1 = outrow * 3 + 1
                    inrow2 = outrow * 3 + 2
                    nincol = nlargecol * 3

                    adder.add(incol0 * nincol + inrow0, 'Llb')
                    adder.add(incol0 * nincol + inrow1, 'Lleft')
                    adder.add(incol0 * nincol + inrow2, 'Llt')
                    adder.add(incol1 * nincol + inrow0, 'Ldown')
                    adder.add(incol1 * nincol + inrow1, 'Lself')
                    adder.add(incol1 * nincol + inrow2, 'Lup')
                    adder.add(incol2 * nincol + inrow0, 'Lrb')
                    adder.add(incol2 * nincol + inrow1, 'Lright')
                    adder.add(incol2 * nincol + inrow2, 'Lrt')

                for i in range(nleft, nlarge):
                    adder.out_index = i

                    outcol = (i - nleft) // base
                    incol0 = outcol * 3
                    incol1 = outcol * 3 + 1
                    incol2 = outcol * 3 + 2
                    outrow = (i - nleft) % base
                    inrow0 = outrow * 3
                    inrow1 = outrow * 3 + 1
                    inrow2 = outrow * 3 + 2
                    ninleft = nleft * 9
                    nincol = base * 3

                    adder.add(incol0 * nincol + inrow0 + ninleft, 'Llb')
                    adder.add(incol0 * nincol + inrow1 + ninleft, 'Lleft')
                    adder.add(incol0 * nincol + inrow2 + ninleft, 'Llt')
                    adder.add(incol1 * nincol + inrow0 + ninleft, 'Ldown')
                    adder.add(incol1 * nincol + inrow1 + ninleft, 'Lself')
                    adder.add(incol1 * nincol + inrow2 + ninleft, 'Lup')
                    adder.add(incol2 * nincol + inrow0 + ninleft, 'Lrb')
                    adder.add(incol2 * nincol + inrow1 + ninleft, 'Lright')
                    adder.add(incol2 * nincol + inrow2 + ninleft, 'Lrt')

                for i in range(nlarge, ntotal):
                    adder.out_index = i

                    outcol = (i - nlarge) // nsmallcol
                    incol0 = outcol * 3
                    incol1 = outcol * 3 + 1
                    incol2 = outcol * 3 + 2
                    outrow = (i - nlarge) % nsmallcol
                    inrow0 = outrow * 3
                    inrow1 = outrow * 3 + 1
                    inrow2 = outrow * 3 + 2
                    ninlarge = nlarge * 9
                    nincol = nsmallcol * 3

                    adder.add(incol0 * nincol + inrow0 + ninlarge, 'Slb')
                    adder.add(incol0 * nincol + inrow1 + ninlarge, 'Sleft')
                    adder.add(incol0 * nincol + inrow2 + ninlarge, 'Slt')
                    adder.add(incol1 * nincol + inrow0 + ninlarge, 'Sdown')
                    adder.add(incol1 * nincol + inrow1 + ninlarge, 'Sself')
                    adder.add(incol1 * nincol + inrow2 + ninlarge, 'Sup')
                    adder.add(incol2 * nincol + inrow0 + ninlarge, 'Srb')
                    adder.add(incol2 * nincol + inrow1 + ninlarge, 'Sright')
                    adder.add(incol2 * nincol + inrow2 + ninlarge, 'Srt')

            else:
                for i in range(nlarge):
                    adder.out_index = i
                    
                    adder.add(i, 'Lself')
            
                for i in range(nlarge, ntotal):
                    adder.out_index = i
                    
                    adder.add(i, 'Sself')
                    
            layer += 1

    return adder.weights, adder.gather_indices, out_layer_conf

def pooling_conv(input, layer_conf, nout, name):
    """
    Create a NN graph convolution kernel as a matrix with shape [Nsens_in x nchannels, Nsens_in, nout]
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    batch_size = input.shape[0]
    nch_in = input.shape[2]

    weights, gather_indices, out_layer_conf = pooling_conv_kernel(input, layer_conf, nout, name)

    nsens_out = 0
    for itype in range(4):
        nsens_out += out_layer_conf[itype] * layer_sizes[itype]

    x = tf.concat([input, tf.constant(np.zeros([batch_size, 1, nch_in], dtype=np.float32))], axis=1)
    x = tf.gather(x, np.reshape(gather_indices, [-1]), axis=1)
    x = tf.reshape(x, [batch_size, nsens_out, -1])
    x = tf.tensordot(x, weights, [[2], [0]])

    return x, out_layer_conf


def pool_z(input, nout):
    """
    Reduce the number of z layers for an input where all z layers are reduced to type 3.
    """

    batch_size = input.shape[0]

    x = tf.reshape(input, [batch_size, -1, layer_sizes[3] * input.shape[-1]])
    x = tf.layers.conv1d(x, nout * layer_sizes[3], [2], strides=[2], padding='same')
    x = tf.reshape(x, [batch_size, -1, nout])

    return x
