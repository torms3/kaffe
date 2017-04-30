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
from DataProvider.python.forward import ForwardScanner

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Forward config.
cfg = config.ForwardConfig(sys.argv[2])

# SET BATCH SIZE HERE
batch_size = int(sys.argv[3]) 

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

# Create scan spec.
scan_spec = dict()
for i in scan_list:
    scan_spec[i] = net.blobs[i].data.shape[-4:]

# Forward scan.
for dataset in dp.datasets:
    idx = dataset.params['dataset_id']
    print 'Forward scan dataset {}'.format(idx)

    # Create ForwardScanner for the current dataset.
    fs = ForwardScanner(dataset, scan_spec, params=scan_params)

    # Scan loop.
    ins = fs.pull_n(batch_size)  # Fetch initial inputs.
  
    inference_start = time.time() 
    count = 0 
    while ins is not None:
        start = time.time()
        count += 1 
        # Set inputs.
        in_shape = None

        for b in xrange(batch_size): 
            if ins[b] != None:
                for k, v in ins[b].iteritems():
                    if in_shape is None:
                        in_shape = (batch_size,) + v.shape
                    net.blobs[k].reshape(*in_shape)
                    net.blobs[k].data[b,...] = v
        # Run forward pass.
        net_start = time.time()
        net.forward()
        print 'Net Time: {}'.format(time.time() - net_start)
        # Extract output data.
        outs = []
 
        for k in scan_spec.iterkeys():
            # doesn't cause index out of bound, and unneeded outputs 
            # are discarded later. 
            for b in xrange(batch_size): 
                outs.append({k: net.blobs[k].data[b,...]})

        fs.push_n(outs)    # Push current outputs.

        ins = fs.pull_n(batch_size)  # Fetch next inputs.
        print 'Elapsed: {}'.format(time.time() - start)

    print 'Inferece time: {}'.format(time.time() - inference_start)
    # Save as file.
    for key in fs.outputs.data.iterkeys():
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx+1, key)
        print 'Save {}...'.format(fname)
        f = h5py.File(fname)
        output = fs.outputs.get_data(key)
        f.create_dataset('/main', data=output)
        f.close()
