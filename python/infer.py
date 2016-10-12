#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser
import caffe
import h5py
import numpy as np
import os
import sys
import time

import config
from DataProvider.python.forward import ForwardScanner
from DataProvider.python.transform import flip, revert_flip

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

# Create data provider.
dp = cfg.get_data_provider(net_spec)

# Scan params.
scan_list   = eval(cfg.get('forward','scan_list'))
scan_params = eval(cfg.get('forward','scan_params'))
save_prefix = cfg.get('forward','save_prefix')
flip_range  = eval(cfg.get('forward','flip_range'))

# Create scan spec.
scan_spec = dict()
for i in scan_list:
    scan_spec[i] = net.blobs[i].data.shape[-4:]

# Forward scan.
for dataset in dp.datasets:
    idx = dataset.dataset_id
    print 'Forward scan dataset {}'.format(idx)

    # Create ForwardScanner for the current dataset.
    accum = ForwardScanner(dataset, scan_spec)
    count = 0.0

    # Inference-time augmentation.
    # Flip augmentation.
    for aug_id in flip_range:
        assert aug_id < 16
        # Convert integer to binary.
        rule = np.array([int(x) for x in bin(aug_id)[2:].zfill(4)])
        print 'Flip augmentation {}'.format(rule)
        # Apply flip transformation.
        for val in dataset._data.itervalues():
            val._data = flip(val._data, rule=rule)

        # Create ForwardScanner for the current augmentation.
        fs = ForwardScanner(dataset, scan_spec, params=scan_params)

        # Scan loop.
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
            # Extract output data.
            outs = dict()
            for k in scan_spec.iterkeys():
                outs[k] = net.blobs[k].data[0,...]
            fs.push(outs)    # Push current outputs.
            # Elapsed time.
            print 'Elapsed: {}'.format(time.time() - start)
            ins = fs.pull()  # Fetch next inputs.

        # Accumulate result.
        for key, val in accum.outputs.data.iteritems():
            print 'Accumulate...'
            output = fs.outputs.get_data(key)
            # Revert output.
            dst = scan_list[key].get('dst', None)
            val._data += revert_flip(output, rule=rule, dst=dst)
            count += 1

        # Revert dataset.
        for val in dataset._data.itervalues():
            val._data = revert_flip(val._data, rule=rule)

    # Save as file.
    for key in accum.outputs.data.iterkeys():
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx+1, key)
        print 'Save {}...'.format(fname)
        f = h5py.File(fname)
        output = accum.outputs.get_data(key)
        output /= count  # Normalize.
        f.create_dataset('/main', data=output)
        f.close()
