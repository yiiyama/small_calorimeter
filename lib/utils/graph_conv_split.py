"""
Generalized 2D convolution where sensor cells are considered as vertices connected to other vertices
by named edges. Edges carry convolution weights.
This implementation: Construct a one sparse tensor that encodes the adjacency per layer. Input for
each layer is separately matmul'ed with the adjacency matrix. Results are concatenated for output.
Also extremely slow.
"""

import tensorflow as tf

NSENSORS = 2679

layer_sizes = [304, 171, 76, 19]
bases = [4, 3, 2, 1]

class AddWeight(object):
    def __init__(self, name, wnames, nout, nin):
        # using identical graph edges at each layers - should check if there will be a difference
        # if we use different weights for different layer types
        w = []
        for iout in range(nout):
            w.append([])
            for iin in range(nin):
                w[iout].append({})
                for wn in wnames:
                    w[iout][iin][wn] = tf.get_variable(
                        '%s_w%d_%s_%s' % (name, iout, iin, wn),
                        shape = [1],
                        trainable = True
                    )

        self.weights = w
        self.nout = nout
        self.nin = nin

        self.out_index = 0
        self.in_layer_size = 0
        self.out_layer_size = 0

        self.indices = dict((wn, []) for wn in wnames)

    def reset(self, in_layer_size, out_layer_size):
        self.in_layer_size = in_layer_size
        self.out_layer_size = out_layer_size
        for key in self.indices:
            del self.indices[key][:]

    def add(self, idx, wname):
        index = (
            self.out_index,
            idx,
        )
        self.indices[wname].append(index)

    def generate(self):
        dense_shape = [self.out_layer_size * self.nout, self.in_layer_size * self.nin]

        x = None

        for iout in range(self.nout):
            for iin in range(self.nin):
                for wn, indices in self.indices.items():
                    if len(indices) == 0:
                        continue

                    full_indices = []
                    for idxout, idxin in indices:
                        full_indices.append((idxout * self.nout + iout, idxin * self.nin + iin))

                    values = tf.tile(self.weights[iout][iin][wn], [len(full_indices)])

                    m = tf.SparseTensor(indices=full_indices, values=values, dense_shape=dense_shape)
                    if x is None:
                        x = m
                    else:
                        x = tf.sparse_add(x, m)

        return tf.sparse_reorder(x)


def nearest_neighbor_conv_kernel(input, layer_conf, nout, name):
    """
    Create a NN graph convolution kernel as a matrix with shape [Nsens_in x nout, Nsens_in x nchannels]
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    wnames = ['Lself', 'LleftL', 'LdownL', 'LrightL', 'LupL', 'Sself', 'SleftS', 'SdownS', 'SrightS', 'SupS', 'LrightS0', 'LrightS1',  'LrightS2', 'LrightS3', 'LupS0', 'LupS1', 'LupS2', 'LupS3', 'SleftL', 'SdownL']

    kernels = []
    adder = AddWeight(name, wnames, nout, input.shape[2])

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
            adder.reset(layer_sizes[itype], layer_sizes[itype])

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

            kernels.append(adder.generate())

            layer += 1

    return kernels

def nearest_neighbor_conv(input, layer_conf, nout, name):
    """
    Perform a NN convolution on input.
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    batch_size = input.shape[0]

    kernels = nearest_neighbor_conv_kernel(input, layer_conf, nout, name)
    bias = tf.get_variable('%s_bias' % name, shape = [nout], trainable = True, initializer = tf.zeros_initializer())

    layer_split = []
    for itype in range(len(layer_conf)):
        layer_split += [layer_sizes[itype]] * layer_conf[itype]

    ys = []

    for layer, y in enumerate(tf.split(input, layer_split, 1)): # [batch, layer sensors, channels]
        print(layer, 'y', y.shape)

        y = tf.reshape(y, [batch_size, -1]) # [batch, layer sensors x channels]
        y = tf.transpose(y, perm=[1, 0]) # [layer sensors x channels, batch]

        print(layer, 'reshape', y.shape)

        y = tf.sparse_tensor_dense_matmul(kernels[layer], y)

        print(layer, 'matmul', y.shape)

        y = tf.transpose(y, perm=[1, 0]) # [batch, layer sensors x channels]
        y = tf.reshape(y, [batch_size, -1, nout]) # [batch, layer sensors, channels]

        print(layer, 'reshape', y.shape)

        ys.append(y)

    x = tf.concat(ys, axis=1) # [batch, sensors, channels]
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.relu(x)

    print('return', x.shape)

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

    out_layer_conf = [0, 0, 0, 0]
    kernels = []
    adder = AddWeight(name, wnames, nout, input.shape[2])

    layer = 0
    adder.in_layer_offset = 0
    adder.out_layer_offset = 0
    for itype in range(4):
        # geometry of the output layer
        if itype == 0:
            outtype = 2
            base = 2
        else:
            outtype = 3
            base = 1

        nlargecol = base * 2
        nleft = nlargecol * base
        nlarge = nleft + base * base
        nsmallcol = base * 4
        nsmall = nsmallcol * nsmallcol
        ntotal = nlarge + nsmall

        # iterate over input layers
        while layer - sum(layer_conf[:itype]) < layer_conf[itype]:
            adder.reset(layer_sizes[itype], layer_sizes[outtype])

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

            kernels.append(adder.generate())
                    
            layer += 1

            out_layer_conf[outtype] += 1

    return kernels, out_layer_conf

def pooling_conv(input, layer_conf, nout, name):
    """
    Create a NN graph convolution kernel as a matrix with shape [Nsens_in x nchannels, Nsens_in, nout]
    @param input      Tensor with shape [batch, sensors, channels]
    @param layer_conf Z-layer configuration of the input [Ntype0, Ntype1, Ntype2, Ntype3]
    @param nout       Number of output channels
    @param name       Variable name prefix
    """

    batch_size = input.shape[0]

    kernels, out_layer_conf = pooling_conv_kernel(input, layer_conf, nout, name)

    layer_split = []
    for itype in range(len(layer_conf)):
        layer_split += [layer_sizes[itype]] * layer_conf[itype]

    ys = []
    for layer, y in enumerate(tf.split(input, layer_split, 1)): # [batch, layer sensors, channels]
        print(layer, 'y', y.shape)

        y = tf.reshape(y, [batch_size, -1]) # [batch, layer sensors x channels]
        y = tf.transpose(y, perm=[1, 0]) # [layer sensors x channels, batch]

        print(layer, 'reshape', y.shape)

        y = tf.sparse_tensor_dense_matmul(kernels[layer], y)

        print(layer, 'matmul', y.shape)

        y = tf.transpose(y, perm=[1, 0]) # [batch, layer sensors x channels]
        y = tf.reshape(y, [batch_size, -1, nout]) # [batch, layer sensors, channels]

        print(layer, 'reshape', y.shape)

        ys.append(y)

    x = tf.concat(ys, axis=1) # [batch, sensors, channels]

    print('return', x.shape)

    return x, out_layer_conf


def pool_z(input, nout):
    """
    Reduce the number of z layers for an input where all z layers are reduced to type 3.
    """

    batch_size = input.shape[0]

    x = tf.reshape(input, [batch_size, -1, layer_sizes[3], input.shape[-1]])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(input, [batch_size * layer_sizes[3], -1, input.shape[-1]])

    x = tf.layers.conv1d(x, nout, [input.shape[-1] * 2], strides=[input.shape[-1] * 2], padding='same')

    x = tf.reshape(x, [batch_size, layer_sizes[3], -1, nout])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [batch_size, -1, nout])

    return x
