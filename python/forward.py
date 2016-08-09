import ConfigParser
import caffe
import h5py
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

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
net_spec   = config.get('forward','net_spec')
dp_params  = config.get('forward','dp_params')
dp = VolumeDataProvider(dspec_path, net_spec, dp_params)

# Forward scan.
scan_spec   = config.get('forward','scan_spec')
save_prefix = config.get('forward','save_prefix')
for idx, dataset in enumerate(dp.datasets):
    print 'Forward scan dataset{}'.format(idx)

    # Scan loop.
    fs  = ForwardScanner(dataset, scan_spec)
    ins = fs.pull()  # Fetch initial inputs.
    while ins is not None:
        start = time.time()
        # Set inputs.
        # TODO(kisuk): VolumeDataLayer?
        # Run forward pass.
        net.forward()
        # Extract output data.
        outs = dict()
        for k, v in scan_spec.iteritems():
            outs[k] = net.blobs[k].data
        fs.push(outs)    # Push current outputs.
        ins = fs.pull()  # Fetch next inputs.
        # Elapsed time.
        print 'Elapsed: {}'.format(time.time() - start)

    # Save as file.
    for name, data in fs.outputs.iteritems():
        fname = '{}_dataset{}_{}.h5'.format(save_prefix, idx, name)
        print 'Save {}...'.format(fname)
        f = h5py.File(fname)
        f.create_dataset('/main', data=data.get_data())
        f.close()
