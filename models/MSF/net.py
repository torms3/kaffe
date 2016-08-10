#!/usr/bin/env python
__doc__ = """

MSF-3D: Multiscale Filter 2D-3D network.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
from caffe import layers as L, params as P
from collections import OrderedDict

def msf(net, lnum, bottom, nout, ks, dilations, lr_mult):
    """
    Create MSF layer.

    Args:
        lnum: Layer number.
        nout: Number of convolution groups.
    """
    # Lambda function for computing crop offset.
    offset = lambda ks, ds, rs: [int((k-1)*(r-d))/2 for k,d,r in zip(ks,ds,rs)]
    # Make sure that the list of dilation factors is in increasing order.
    dilations = sorted(dilations)
    # Convolution w/ different dilation factors.
    for d in dilations:
        lname = 'conv{}s{}'.format(lnum, d)
        # Convolution w/ anasotropic dilation [1,d,d].
        net[lname] = L.Convolution(bottom,
            kernel_size=ks, dilation=[1,d,d], num_output=nout,
            weight_filler=dict(type="msra"), bias_filler=dict(type="constant"),
            param=[dict(lr_mult=lr_mult), dict(lr_mult=lr_mult)])
    # ConvolutionLayer w/ the largest dilation becomes crop reference.
    d0 = dilations[-1]
    ref = net['conv{}s{}'.format(lnum, d0)]
    # List of bottoms as an input to ConcatLayer.
    bottoms = [ref]
    for d in dilations[:-1]:
        iname = 'conv{}s{}'.format(lnum, d)
        oname = 'crop{}s{}'.format(lnum, d)
        top = L.Crop(net[iname], ref, offset=offset(ks,[1,d,d],[1,d0,d0]))
        net[oname] = top
        bottoms.append(top)
    # Concat along the channel axis.
    concat = L.Concat(*bottoms, axis=1)
    return concat, L.ReLU(concat, in_place=True)


def conv_relu(bottom, nout, ks, lr_mult, d=1, s=1):
    """Create convolution layer w/ ReLU activation."""
    conv = L.Convolution(bottom,
        num_output=nout, kernel_size=ks, dilation=d, stride=s,
        weight_filler=dict(type="msra"), bias_filler=dict(type="constant"),
        param=[dict(lr_mult=lr_mult), dict(lr_mult=lr_mult)])
    return conv, L.ReLU(conv, in_place=True)


def net_spec(outsz):
    """Create net specification given outsz."""
    fov     = [9,97,97]
    insz    = [x + y - 1 for x, y in zip(outsz,fov)]
    in_dim  = [1,1] + insz
    out_dim = [1,3] + outsz
    spec = {'input':in_dim, 'label':out_dim, 'label_mask':out_dim}
    return OrderedDict(sorted(spec.items(), key=lambda x: x[0]))


def multiscale_filter(outsz, phase):

    # Net specification.
    n = caffe.NetSpec()
    spec = net_spec(outsz)

    # Data layers.
    assert phase in ['train','val','deploy']
    if phase == 'deploy':
        n['input'] = L.Input(shape=dict(dim=spec['input']))
    else:
        for k, v in spec.iteritems():
            n[k] = L.Input(shape=dict(dim=v))

    # Patch averaging.
    lr_mult = 1.0/(outsz[0]*outsz[1]*outsz[2])

    # The net itself.
    n.conv1a, n.relu1a = conv_relu(n['input'], 64, [1,5,5], lr_mult)
    n.conv1b, n.relu1b = conv_relu(n.relu1a,   64, [1,5,5], lr_mult)

    # Params for multiscale filter layers.
    ds = [1,2,3,4]  # Dilations for each convolution group.
    ns = 16         # Number of convolution groups.

    # 2D multiscale filter layers.
    ks = [1,3,3]
    n.concat2, n.relu2 = msf(n, 2, n.relu1b, ns, ks, ds, lr_mult)
    n.concat3, n.relu3 = msf(n, 3, n.relu2,  ns, ks, ds, lr_mult)
    n.concat4, n.relu4 = msf(n, 4, n.relu3,  ns, ks, ds, lr_mult)

    # 3D multiscale filter layers.
    ks = [2,3,3]
    n.concat5,  n.relu5  = msf(n, 5,  n.relu4,  ns, ks, ds, lr_mult)
    n.concat6,  n.relu6  = msf(n, 6,  n.relu5,  ns, ks, ds, lr_mult)
    n.concat7,  n.relu7  = msf(n, 7,  n.relu6,  ns, ks, ds, lr_mult)
    n.concat8,  n.relu8  = msf(n, 8,  n.relu7,  ns, ks, ds, lr_mult)
    n.concat9,  n.relu9  = msf(n, 9,  n.relu8,  ns, ks, ds, lr_mult)
    n.concat10, n.relu10 = msf(n, 10, n.relu9,  ns, ks, ds, lr_mult)
    n.concat11, n.relu11 = msf(n, 11, n.relu10, ns, ks, ds, lr_mult)
    n.concat12, n.relu12 = msf(n, 12, n.relu11, ns, ks, ds, lr_mult)

    # Classification layer.
    n.convx, n.relux = conv_relu(n.relu12, 200, [1,1,1], lr_mult)

    # Loss layer.
    if phase == 'deploy':
        n.output, _ = conv_relu(n.output, 3, [1,1,1], lr_mult)
        n.sigmoid = L.Sigmoid(n.output, in_place=True)
    else:
        n.drop = L.Dropout(n.relux, in_place=True)
        n.output, _ = conv_relu(n.drop, 3, [1,1,1], lr_mult)
        # Custom python loss layer.
        pylayer = 'SigmoidCrossEntropyLossLayer'
        bottoms = [n.output, n['label'], n['label_mask']]
        n.loss, n.loss2, n.cerr = L.Python(*bottoms,
            module='volume_loss_layers', layer=pylayer,
            ntop=3, loss_weight=[1,0,0])

    return n.to_proto()


def make_net(outsz):
    # Train.
    phase = 'train'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(multiscale_filter(outsz, phase)))

    # Validation.
    phase = 'val'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(multiscale_filter(outsz, phase)))

    # Deploy.
    phase = 'deploy'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(multiscale_filter(outsz, phase)))


if __name__ == '__main__':

    from sys import argv

    if len(argv)==2:
        make_net(list(eval(argv[1])))
    if len(argv)==4:
        make_net([int(argv[1]),int(argv[2]),int(argv[3])])
    else:
        print 'Usage: [python net.py z,y,x] or '
        print '       [python net.py z y x]'
