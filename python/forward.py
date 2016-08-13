#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser
import caffe
import h5py
import os
import sys
import time

import config
from DataProvider.python.data_provider import VolumeDataProvider
from DataProvider.python.forward import ForwardScanner

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Forward config.
cfg = config.ForwardConfig(sys.argv[2])

# Create an inference net.
net = cfg.net()

# Create net spec.
net_spec = dict()
for i in net.inputs:
    net_spec[i] = net.blobs[i].data.shape[-3:]

# Create VolumeDataProvider.
dp = cfg.get_data_provider(net_spec)

# Scan params.
scan_list   = eval(cfg.get('forward','scan_list'))
scan_params = eval(cfg.get('forward','scan_params'))
save_prefix = cfg.get('forward','save_prefix')

# Create scan spec.
scan_spec = dict()
for i in scan_list:
    scan_spec[i] = net.blobs[i].data.shape[-4:]

# Forward scan.
for dataset in dp.datasets:
    idx = dataset.dataset_id
    print 'Forward scan dataset{}'.format(idx)

    # Scan loop.
    fs  = ForwardScanner(dataset, scan_spec, params=scan_params)
    ins = fs.pull()  # Fetch initial inputs.
    while ins is not None:
        start = time.time()
        # Set inputs.
        for k, v in ins.iteritems():
            shape = (1,) + v.shape
            net.blobs[k].reshape(*shape)
            net.blobs[k].data[0,...] = v
        # Run forward pass.
        net.forward()
        # Elapsed time.
        print 'Elapsed: {}'.format(time.time() - start)
        # Extract output data.
        outs = dict()
        for k in scan_spec.iterkeys():
            outs[k] = net.blobs[k].data[0,...]
        fs.push(outs)    # Push current outputs.
        ins = fs.pull()  # Fetch next inputs.

    # Save as file.
    for name, data in fs.outputs.iteritems():
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx, name)
        print 'Save {}...'.format(fname)
        f = h5py.File(fname)
        f.create_dataset('/main', data=data.get_data())
        f.close()
