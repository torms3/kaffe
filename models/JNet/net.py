#!/usr/bin/env python
__doc__ = """

JNet: Jonathan Zung's network similar to U-Net.

Jonathan Zung <jzung@princeton.edu>,
Jingpeng Wu <jingpeng.wu@gmail.com>,
Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
from caffe import layers as L, params as P
from collections import OrderedDict

nfeatures = [
    [1,24,28,32,48,64],
    [0,12,24,28,32,48],
    [0, 0,12,24,28,32],
    [0, 0, 0,12,24,28],
    [0, 0, 0, 0,12,24],
    [0, 0, 0, 0, 0, 3],
]

sizes = [[1,4,4],
         [1,4,4],
         [2,4,4],
         [4,4,4],
         [8,4,4]]


strides = [[1,2,2],
           [1,2,2],
           [1,2,2],
           [1,2,2],
           [1,2,2]]


activations = [[lambda x: L.ELU(x,in_place=True) for i in l] for l in nfeatures]
# Don't apply activation function for the final layer, so that we can compute
# cross-entropy from logits.
activations[-1][-1] = lambda x: x


def up(bottoms, num_output, ks, s, lr_mult=1, bias_term=False):
    """Implement convolution/downsample."""
    # TODO(kisuk): Factorizing 3D convolution.
    param = [dict(lr_mult=lr_mult)]
    if bias_term:
        param.append(dict(lr_mult=lr_mult))
    return L.Convolution(bottoms,
		num_output=num_output, kernel_size=ks, stride=s,
		weight_filler=dict(type="msra"), bias_filler=dict(type="constant"),
        param=param, bias_term=bias_term)


def down(bottoms, num_output, ks, s, lr_mult=1):
    """Implement deconvolution/upsample."""
    # TODO(kisuk): Factorizing 3D deconvolution.
    return L.Deconvolution(bottoms,
        convolution_param=dict(
			num_output=num_output, kernel_size=ks, stride=s,
			weight_filler=dict(type="msra"), bias_filler=dict(type="constant")),
        param=[dict(lr_mult=lr_mult), dict(lr_mult=lr_mult)])


def is_valid(i,j):
    return (i<=j and i>=0 and j>=0 and
        i<len(nfeatures) and j<len(nfeatures[i]) and nfeatures[i][j]>0)


def forward(net, bottom, lr_mult=1):
    """
    TODO(kisuk): Documentation.
    """
    is_input     = lambda i, j: i==0 and j==0
    is_output    = lambda i, j: i+1==len(nfeatures) and j+1==len(nfeatures[0])
    has_feedback = lambda i, j: is_valid(i-1, j)
    has_selfloop = lambda i, j, x=nfeatures: is_valid(i-1, j-1) and x[i][j]==x[i-1][j-1]

    tops = [[0 for j in i] for i in nfeatures]
    tops[0][0] = bottom

    for i in xrange(len(nfeatures)):  # Time steps.
        for j in xrange(i, len(nfeatures[i])):  # Layers.
            if is_input(i,j):
                name = 'conv{},{}'.format(i,j+1)
                prev = up(tops[i][j], nfeatures[i][j+1], sizes[j-i], strides[j-i], bias_term=True)
                net[name] = prev
                tops[i][j+1] = prev
            elif is_output(i,j):
                net['output'] = down(tops[i-1][j], nfeatures[i][j], sizes[j-i], strides[j-i])
            elif is_valid(i,j):
                postfix = '{},{}'.format(i,j)
                bottoms = list() if tops[i][j]==0 else [tops[i][j]]
                # Top-down feedback connection from the previous time step.
                if has_feedback(i,j):
                    prev = down(tops[i-1][j], nfeatures[i][j], sizes[j-i], strides[j-i])
                    net['deconv'+postfix] = prev
                    bottoms.append(prev)
                # Self-loop connection from the previous time step.
                if has_selfloop(i,j):
                    prev = tops[i-1][j-1]
                    bottoms.append(prev)
                # Sum, add biases, and activate.
                assert len(bottoms) > 0
                if len(bottoms) > 1:
                    # Sum, if needed.
                    prev = L.Eltwise(*bottoms)
                    net['sum'+postfix] = prev
                else:
                    prev = bottoms[0]
                # Activate.
                prev = activations[i][j](prev)
                net['relu'+postfix] = prev
                # Replace bottoms w/ a resulting top.
                tops[i][j] = prev
                # Propagates down to the next layer.
                if is_valid(i,j+1):
                    name = 'conv{},{}'.format(i,j+1)
                    b = True if i==0 else False
                    prev = up(tops[i][j], nfeatures[i][j+1], sizes[j-i], strides[j-i], bias_term=b)
                    net[name] = prev
                    tops[i][j+1] = prev


def net_spec(outsz):
    in_dim  = [1,1] + outsz
    out_dim = [1,3] + outsz
    spec	= {'input':in_dim, 'label':out_dim, 'label_mask':out_dim}
    return OrderedDict(sorted(spec.items(), key=lambda x: x[0]))


def jnet(outsz, phase):

    # Net specification.
    n = caffe.NetSpec()
    spec = net_spec(outsz)

    # Data layers.
    assert phase in ['train', 'val', 'deploy']
    if phase == 'deploy':
        n['input'] = L.Input(shape=dict(dim=spec['input']))
    else:
        for k, v in spec.iteritems():
            n[k] = L.Input(shape=dict(dim=v))

    # The net itself.
    forward(n, n['input'])

    # Loss layer.
    if phase == 'deploy':
        n.sigmoid = L.Sigmoid(n['output'], in_place=True)
    else:
        # Custom python loss layer.
        pylayer = 'SigmoidCrossEntropyLossLayer'
        bottoms = [n['output'], n['label'], n['label_mask']]
        n.loss, n.loss2, n.cerr = L.Python(*bottoms,
            module='volume_loss_layers', layer=pylayer,
            ntop=3, loss_weight=[1,0,0])

    return n.to_proto()


def make_net(outsz):
    # Train.
    phase = 'train'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(jnet(outsz, phase)))

    # Validation.
    phase = 'val'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(jnet(outsz, phase)))

    # Deploy.
    phase = 'deploy'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(jnet(outsz, phase)))


if __name__ == '__main__':

    from sys import argv

    if len(argv) == 2:
        make_net(list(eval(argv[1])))
    if len(argv) == 4:
        make_net([int(argv[1]), int(argv[2]), int(argv[3])])
    else:
        print 'Usage: 	[python net.py z,y,x] or '
        print '			[python net.py z y x]'
