import ConfigParser
import caffe
import h5py
import os
import sys
import time

import parser
from data_provider.python.data_provider import VolumeDataProvider
from data_provider.python.forward import ForwardScanner

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Config.
config = ConfigParser.ConfigParser()
config.read(sys.argv[2])

# Create an inference net.
model   = config.get('forward','model')
weights = config.get('forward','weights')
net = caffe.Net(model, weights, caffe.TEST)

# Create VolumeDataProvider.
dspec_path = config.get('forward','data_spec')
net_spec   = eval(config.get('forward','net_spec'))
dp_params  = eval(config.get('forward','dp_params'))
dp = VolumeDataProvider(dspec_path, net_spec, dp_params)

# Forward scan.
scan_spec   = eval(config.get('forward','scan_spec'))
scan_params = eval(config.get('forward','scan_params'))
save_prefix = config.get('forward','save_prefix')
for idx, dataset in enumerate(dp.datasets):
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
