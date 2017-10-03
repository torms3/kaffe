#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import shutil
import ConfigParser
import h5py
import os
import sys
import time

import config
from DataProvider.python.forward import ForwardScanner

# Forward config.
cfg = config.ForwardConfig(sys.argv[1])

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
#THIS NEEDS TO BE CHANGED FOR EVERY NET UGLY HACK BOHAHA
scan_spec = {'output': (3, 14, 180, 180)}

# Forward scan.
for dataset in dp.datasets:
    
    idx = dataset.params['dataset_id']
    print 'Forward scan dataset {}'.format(idx)

    # Create ForwardScanner for the current dataset.
    import pdb; pdb.set_trace() 
    fs = ForwardScanner(dataset, scan_spec, params=scan_params)

    # Scan loop.
    ins = fs.pull()  # Fetch initial inputs.
    while ins is not None:
        start = time.time()
        # Run forward pass.
        outs = dict()
        outs["output"] = net.forward(ins["input"])[0][...]
        # Extract output data.
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

        try:
           print ("Deleting temp folder...")
           shutil.rmtree('/tmp/{}'.format(os.getpid())); sys.exit()
	except:
	   print ("Couldn't remove the temp folder.")
        else:
	   print ("Temp folder deleted.")

