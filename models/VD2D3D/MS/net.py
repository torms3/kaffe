#!/usr/bin/env python
__doc__ = """

VD2D3D-MS: Very Deep Multiscale 2D-3D ConvNet.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
from caffe import layers as L, params as P
from collections import OrderedDict

def conv_relu(bottom, nout, ks, lr_mult, d=1, s=1, share=None):
    """Create convolution layer w/ ReLU activation."""
    if share is None:
        param = [dict(lr_mult=lr_mult), dict(lr_mult=lr_mult)]
    else:
        param = [dict(name=share+"_w", lr_mult=lr_mult),
                 dict(name=share+"_b", lr_mult=lr_mult)]
    conv = L.Convolution(bottom,
        num_output=nout, kernel_size=ks, dilation=d, stride=s,
        weight_filler=dict(type="msra"), bias_filler=dict(type="constant"),
        param=param)
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks, d=1, s=1):
    return L.Pooling(bottom,
        pool=P.Pooling.MAX, kernel_size=ks, dilation=d, stride=s)


def net_spec(outsz):
    """Create net specification given outsz."""
    fov     = [9,109,109]
    insz    = [x + y - 1 for x, y in zip(outsz,fov)]
    in_dim  = [1,1] + insz
    out_dim = [1,3] + outsz
    spec = {'input':in_dim, 'label':out_dim, 'label_mask':out_dim}
    return OrderedDict(sorted(spec.items(), key=lambda x: x[0])), fov


def vd2d3d(outsz, phase):

    # Net specification.
    n = caffe.NetSpec()
    spec, fov = net_spec(outsz)

    # Data layers.
    assert phase in ['train','val','deploy']
    if phase == 'deploy':
        n['input'] = L.Input(shape=dict(dim=spec['input']))
    else:
        for k, v in spec.iteritems():
            n[k] = L.Input(shape=dict(dim=v))

    # Patch averaging.
    lr_mult = 1.0/(outsz[0]*outsz[1]*outsz[2])

    """
    P3 path.
    """
    # Crop input.
    fov_p3 = [5,109,109]
    offset = [int(x-y)/2 for x,y in zip(fov,fov_p3)]
    size   = [int(x-2*y) for x,y in zip(fov,offset)]
    n.ref_p3  = L.DummyData(shape=dict(dim=size))
    n.data_p3 = L.Crop(net['input'], n.ref_p3, offset=offset)

    # P3 branch.
    n.conv1a_p3, n.relu1a_p3 = conv_relu(n.data_p3,   24, [1,3,3], lr_mult, share="conv1a")
    n.conv1b_p3, n.relu1b_p3 = conv_relu(n.relu1a_p3, 24, [1,3,3], lr_mult, share="conv1b")
    n.conv1c_p3, n.relu1c_p3 = conv_relu(n.relu1b_p3, 24, [1,2,2], lr_mult, share="conv1c")
    n.pool1_p3 = max_pool(n.relu1c_p3, [1,2,2])

    # Anisotropic dilation.
    d = [1,2,2]

    n.conv2a_p3, n.relu2a_p3 = conv_relu(n.pool1_p3,  36, [1,3,3], lr_mult, d=d, share="conv2a")
    n.conv2b_p3, n.relu2b_p3 = conv_relu(n.relu2a_p3, 36, [1,3,3], lr_mult, d=d, share="conv2b")
    n.pool2_p3 = max_pool(n.relu2b_p3, [1,2,2], d=d)

    # Anisotropic dilation.
    d = [1,4,4]

    n.conv3a_p3, n.relu3a_p3 = conv_relu(n.pool2_p3,  48, [1,3,3], lr_mult, d=d, share="conv3a")
    n.conv3b_p3, n.relu3b_p3 = conv_relu(n.relu3a_p3, 48, [1,3,3], lr_mult, d=d, share="conv3b")
    n.pool3_p3 = max_pool(n.relu3b_p3, [1,2,2], d=d)

    # Anisotropic dilation.
    d = [1,8,8]

    n.conv4a_p3, n.relu4a_p3 = conv_relu(n.pool3_p3,  60, [1,3,3], lr_mult, d=d)
    n.conv4b_p3, n.relu4b_p3 = conv_relu(n.relu4a_p3, 60, [2,3,3], lr_mult, d=d)
    n.pool4_p3 = max_pool(n.relu4b_p3, [2,2,2], d=d)

    n.conv5a_p3, n.relu5a_p3 = conv_relu(n.pool4_p3,  60, [2,3,3], lr_mult, d=d)
    n.conv5b_p3, n.relu5b_p3 = conv_relu(n.relu5a_p3, 60, [2,3,3], lr_mult, d=d)

    """
    P2 path.
    """
    # Crop input.
    fov_p2 = [7,73,73]
    offset = [int(x-y)/2 for x,y in zip(fov,fov_p2)]
    size   = [int(x-2*y) for x,y in zip(fov,offset)]
    n.ref_p2  = L.DummyData(shape=dict(dim=size))
    n.data_p2 = L.Crop(net['input'], n.ref_p2, offset=offset)

    # P2 branch.
    n.conv1a_p2, n.relu1a_p2 = conv_relu(n.data_p2,   24, [1,3,3], lr_mult, share="conv1a")
    n.conv1b_p2, n.relu1b_p2 = conv_relu(n.relu1a_p2, 24, [1,3,3], lr_mult, share="conv1b")
    n.conv1c_p2, n.relu1c_p2 = conv_relu(n.relu1b_p2, 24, [1,2,2], lr_mult, share="conv1c")
    n.pool1_p2 = max_pool(n.relu1c_p2, [1,2,2])

    # Anisotropic dilation.
    d = [1,2,2]

    n.conv2a_p2, n.relu2a_p2 = conv_relu(n.pool1_p2,  36, [1,3,3], lr_mult, d=d, share="conv2a")
    n.conv2b_p2, n.relu2b_p2 = conv_relu(n.relu2a_p2, 36, [1,3,3], lr_mult, d=d, share="conv2b")
    n.pool2_p2 = max_pool(n.relu2b_p2, [1,2,2], d=d)

    # Anisotropic dilation.
    d = [1,4,4]

    n.conv3a_p2, n.relu3a_p2 = conv_relu(n.pool2_p2,  48, [1,3,3], lr_mult, d=d, share="conv3a")
    n.conv3b_p2, n.relu3b_p2 = conv_relu(n.relu3a_p2, 48, [1,3,3], lr_mult, d=d, share="conv3b")
    n.pool3_p2 = max_pool(n.relu3b_p2, [2,2,2], d=d)

    n.conv4a_p2, n.relu4a_p2 = conv_relu(n.pool3_p2,  48, [2,3,3], lr_mult, d=d)
    n.conv4b_p2, n.relu4b_p2 = conv_relu(n.relu4a_p2, 48, [2,3,3], lr_mult, d=d)
    n.pool4_p2 = max_pool(n.relu4b_p2, [2,2,2], d=d)

    n.conv5a_p2, n.relu5a_p2 = conv_relu(n.pool4_p2,  60, [2,3,3], lr_mult, d=d)
    n.conv5b_p2, n.relu5b_p2 = conv_relu(n.relu5a_p2, 60, [2,3,3], lr_mult, d=d)

    """
    P1 path.
    """
    # Crop input.
    fov_p1 = [9,45,45]
    offset = [int(x-y)/2 for x,y in zip(fov,fov_p1)]
    size   = [int(x-2*y) for x,y in zip(fov,offset)]
    n.ref_p1  = L.DummyData(shape=dict(dim=size))
    n.data_p1 = L.Crop(net['input'], n.ref_p1, offset=offset)

    # P1 branch.
    n.conv1a_p1, n.relu1a_p1 = conv_relu(n.data_p1,   24, [1,3,3], lr_mult, share="conv1a")
    n.conv1b_p1, n.relu1b_p1 = conv_relu(n.relu1a_p1, 24, [1,3,3], lr_mult, share="conv1b")
    n.conv1c_p1, n.relu1c_p1 = conv_relu(n.relu1b_p1, 24, [1,2,2], lr_mult, share="conv1c")
    n.pool1_p1 = max_pool(n.relu1c_p1, [1,2,2])

    # Anisotropic dilation.
    d = [1,2,2]

    n.conv2a_p1, n.relu2a_p1 = conv_relu(n.pool1_p1,  36, [1,3,3], lr_mult, d=d, share="conv2a")
    n.conv2b_p1, n.relu2b_p1 = conv_relu(n.relu2a_p1, 36, [1,3,3], lr_mult, d=d, share="conv2b")
    n.pool2_p1 = max_pool(n.relu2b_p1, [1,2,2], d=d)

    n.conv3a_p1, n.relu3a_p1 = conv_relu(n.pool2_p1,  36, [2,3,3], lr_mult, d=d)
    n.conv3b_p1, n.relu3b_p1 = conv_relu(n.relu3a_p1, 36, [2,3,3], lr_mult, d=d)
    n.pool3_p1 = max_pool(n.relu3b_p1, [2,2,2], d=d)

    n.conv4a_p1, n.relu4a_p1 = conv_relu(n.pool3_p1,  48, [2,3,3], lr_mult, d=d)
    n.conv4b_p1, n.relu4b_p1 = conv_relu(n.relu4a_p1, 48, [2,3,3], lr_mult, d=d)
    n.pool4_p1 = max_pool(n.relu4b_p1, [2,2,2], d=d)

    n.conv5a_p1, n.relu5a_p1 = conv_relu(n.pool4_p1,  60, [2,3,3], lr_mult, d=d)
    n.conv5b_p1, n.relu5b_p1 = conv_relu(n.relu5a_p1, 60, [2,3,3], lr_mult, d=d)

    """
    Path integration.
    """
    bottoms  = [n.conv5b_p1, n.conv5b_p2, n.conv5b_p3]
    n.concat = L.Concat(*bottoms, axis=1)
    n.convx, n.relux = conv_relu(n.concat, 200, [1,1,1], lr_mult)

    # Loss layer.
    if phase == 'deploy':
        n.output, _ = conv_relu(n.relux, 3, [1,1,1], lr_mult)
        n.sigmoid = L.Sigmoid(n.output, in_place=True)
    else:
        n.dropx = L.Dropout(n.relu5b)
        n.output, _ = conv_relu(n.dropx, 3, [1,1,1], lr_mult)
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
        f.write(str(vd2d3d(outsz, phase)))

    # Validation.
    phase = 'val'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(vd2d3d(outsz, phase)))

    # Deploy.
    phase = 'deploy'
    fname = '{}.prototxt'.format(phase)
    with open(fname, 'w') as f:
        f.write(str(vd2d3d(outsz, phase)))


if __name__ == '__main__':

    from sys import argv

    if len(argv)==2:
        make_net(list(eval(argv[1])))
    if len(argv)==4:
        make_net([int(argv[1]),int(argv[2]),int(argv[3])])
    else:
        print 'Usage: [python net.py z,y,x] or '
        print '       [python net.py z y x]'
