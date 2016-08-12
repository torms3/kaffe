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
    [1, 24,28,32,48, 64],
    [0, 12,24,28,32, 48],
    [0, 0, 12,24,28, 32],
    [0, 0, 0, 12,24, 28],
    [0, 0, 0, 0, 12, 24],
    [0, 0, 0, 0,  0,  3],
]

sizes = [[4,4,1],
         [4,4,1],
         [4,4,2],
         [4,4,4],
         [4,4,8]]

strides = [[2,2,1],
           [2,2,1],
           [2,2,1],
           [2,2,1],
           [2,2,1]]

activations = [[lambda x: L.ELU(x,in_place=True) for i in l] for l in nfeatures]
# Don't apply activation function for the final layer, so that we can compute
# cross-entropy from logits.
activations[-1][-1] = lambda x: x

def up(bottom, num_output, ks, s, lr_mult=1, bias_term=False):
    """Implement convolution/downsample."""
    # TODO(kisuk): Factorizing 3D convolution.
    return L.Convolution(bottom,
				num_output=num_output, kernel_size=ks, stride=s,
				weight_filler=dict(type="msra"), param=dict(lr_mult=lr_mult),
                bias_term=bias_term)


def down(bottom, num_output, ks, s, lr_mult=1, bias_term=False):
    """Implement deconvolution/upsample."""
    # TODO(kisuk): Factorizing 3D deconvolution.
    return L.Deconvolution(bottom,
				num_output=num_output, kernel_size=ks, stride=s,
				weight_filler=dict(type="msra"), param=dict(lr_mult=lr_mult),
                bias_term=bias_term)


def forward(net, bottom, lr_mult=1):
    """
    TODO(kisuk): Documentation.
    """
    is_input     = lambda i, j: i==0 and j==0
    is_output    = lambda i, j: i-1==len(nfeatures) and j-1==len(nfeatures[0])
    is_valid     = lambda i, j, x=nfeatures: i<=j and x[i][j]>0
    has_feedback = lambda i, j: is_valid(i-1, j)
    has_selfloop = lambda i, j, x=nfeatures: is_valid(i-1, j-1) and x[i][j]==x[i-1][j-1]

    tops = [[list() for j in i] for i in nfeatures]

    for i in xrange(len(nfeatures)):  # Time steps.
        for j in xrange(i, len(nfeatures[i])):  # Layers.
            if is_input(i,j):
                continue
            elif is_output(i,j):
                net['output'] = down(tops[i-1][j], nfeatures[i][j], sizes[j-i], strides[j-i], lr_mult, bias_term=True)
            elif is_valid(i,j):
                postfix = '{},{}'.format(i,j)
                # Top-down feedback connection from the previous time step.
                if has_feedback(i,j):
                    prev = down(tops[i-1][j], nfeatures[i][j], sizes[j-i], strides[j-i], lr_mult)
                    net['deconv'+postfix] = prev
                    tops[i][j].append(prev)
                # Self-loop connection from the previous time step.
                if has_selfloop(i,j):
                    prev = tops[i-1][j-1]
                    tops[i][j].append(prev)
                # Sum, add biases, and activate.
                if len(tops[i][j]) > 1:
                    # Sum, if needed.
                    prev = L.Eltwise(*tops[i][j])
                    net['sum'+postfix] = prev
                else:
                    prev = tops[i][j]
                # Add biases.
                # prev = L.Bias(...)
                # net['bias'+postfix] = prev
                # Activate.
                prev = activations(prev)
                net['relu'+postfix] = prev
                # Replace bottoms w/ a resulting top.
                tops[i][j] = [prev]
                # Propagates down to the next layer.
                if is_valid(i,j+1):
                    name = 'conv{},{}'.format(i,j+1)
                    net[name] = up(prev, nfeatures[i][j+1], sizes[j-i], strides[j-i], lr_mult)


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

    # Patch averaging.
    lr_mult = 1.0/(outsz[0]*outsz[1]*outsz[2])

    # The net itself.
    forward(n, n['input'], lr_mult=lr_mult)

    return n.to_proto()


def make_net(outsz):
	# Train
	phase = 'train'
	fname = '{}.prototxt'.format(phase)
	with open(fname, 'w') as f:
		f.write(str(jnet(outsz, phase)))

	# Validation
	phase = 'val'
	fname = '{}.prototxt'.formate(phase)
	with open(fname, 'w') as f:
		f.write(str(jnet(outsz, phase)))

	# Benchmark
	phase = 'benchmark'
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
