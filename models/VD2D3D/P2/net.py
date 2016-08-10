#!/usr/bin/env python
__doc__ = """

VD2D3D-P1: Very Deep 2D-3D ConvNet with one downsampling layer.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

def conv_relu(bottom, nout, ks, lr_mult, d=1, s=1):
    """Create convolution layer w/ ReLU activation."""
    conv = L.Convolution(bottom,
        num_output=nout, kernel_size=ks, dilation=d, stride=s,
        weight_filler=dict(type="msra"), bias_filler=dict(type="constant"),
        param=[dict(lr_mult=lr_mult), dict(lr_mult=lr_mult)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks, d=1, s=1):
    return L.Pooling(bottom,
        pool=P.Pooling.MAX, kernel_size=ks, dilation=d, stride=s)


def net_spec(outsz):
    """Create net specification given outsz."""
    fov     = [7,73,73]
    insz    = [x + y - 1 for x, y in zip(outsz,fov)]
    in_dim  = [1,1] + insz
    out_dim = [1,3] + outsz
    spec = {'input':in_dim, 'label':out_dim, 'label_mask':out_dim}
    return OrderedDict(sorted(spec.items(), key=lambda x: x[0]))


def vd2d3d(outsz):

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
    n.conv1a, n.relu1a = conv_relu(n['input'], 24, [1,3,3], lr_mult)
    n.conv1b, n.relu1b = conv_relu(n.relu1a,   24, [1,3,3], lr_mult)
    n.conv1c, n.relu1c = conv_relu(n.relu1b,   24, [1,2,2], lr_mult)
    n.pool1 = max_pool(n.relu1c, [1,2,2])

    # Anisotropic dilation.
    d = [1,2,2]

    n.conv2a, n.relu2a = conv_relu(n.pool1,  36, [1,3,3], lr_mult, d=d)
    n.conv2b, n.relu2b = conv_relu(n.relu2a, 36, [1,3,3], lr_mult, d=d)
    n.pool2 = max_pool(n.relu2b, [1,2,2], d=d)

    # Anisotropic dilation.
    d = [1,4,4]

    n.conv3a, n.relu3a = conv_relu(n.pool2,  48, [1,3,3], lr_mult, d=d)
    n.conv3b, n.relu3b = conv_relu(n.relu3a, 48, [1,3,3], lr_mult, d=d)
    n.pool3 = max_pool(n.relu3b, [2,2,2], d=d)

    n.conv4a, n.relu4a = conv_relu(n.pool3,  60, [2,3,3], lr_mult, d=d)
    n.conv4b, n.relu4b = conv_relu(n.relu4a, 60, [2,3,3], lr_mult, d=d)
    n.pool4 = max_pool(n.relu4b, [2,2,2], d=d)

    n.conv5a, n.relu5a = conv_relu(n.pool4,  60, [2,3,3], lr_mult, d=d)
    n.conv5b, n.relu5b = conv_relu(n.relu5a,100, [2,3,3], lr_mult, d=d)

    # Loss layer.
    if phase == 'deploy':
        n.output, _ = conv_relu(n.relu5b, 3, [1,1,1], lr_mult)
        n.sigmoid = L.Sigmoid(n.output, in_place=True)
    else:
        n.drop5b = L.Dropout(n.relu5b)
        n.output, _ = conv_relu(n.drop5b, 3, [1,1,1], lr_mult)
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
