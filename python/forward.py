#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import ConfigParser
import h5py
import os
import sys
import time

import config
from DataProvider.python.forward import ForwardScanner
# Forward config.
cfg = config.ForwardConfig(sys.argv[2])

# Create an inference net.
net = cfg.net()

# Create net spec.
net_spec = dict()
'''for i in net.inputs:
    net_spec[i] = net.blobs[i].data.shape[-3:]'''
net_spec = {'input': (18, 192, 192)}
# Create data provider.
dp = cfg.get_data_provider(net_spec)

# Scan params.
scan_list   = eval(cfg.get('forward','scan_list'))
scan_params = eval(cfg.get('forward','scan_params'))
save_prefix = cfg.get('forward','save_prefix')

# Create scan spec.
scan_spec = dict()
scan_spec = {'output': (3, 18, 192, 192)}

# Forward scan.
for dataset in dp.datasets:
    idx = dataset.params['dataset_id']
    print 'Forward scan dataset {}'.format(idx)

    # Create ForwardScanner for the current dataset.
    fs = ForwardScanner(dataset, scan_spec, params=scan_params)

    # Scan loop.
    ins = fs.pull()  # Fetch initial inputs.
    while ins is not None:
        start = time.time()
        # Set inputs.
        '''for k, v in ins.iteritems():
            shape = (1,) + v.shape
            net.blobs[k].reshape(*shape)
            net.blobs[k].data[0,...] = v'''
        # Run forward pass.
        outs = dict()
        outs["output"] = net.forward(ins["input"])[0][...]
        # Extract output data.
        '''outs = dict()
        for k in scan_spec.iterkeys():
            outs[k] = net.blobs[k].data[0,...]'''
        fs.push(outs)    # Push current outputs.
        # Elapsed time.
        print 'Elapsed: {}'.format(time.time() - start)
        ins = fs.pull()  # Fetch next inputs.

    # Save as file.
    for key in fs.outputs.data.iterkeys():
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx+1, key)
        print 'Save {}...'.format(fname)
        f = h5py.File(fname)
        output = fs.outputs.get_data(key)
        f.create_dataset('/main', data=output)
        f.close()
