#!/usr/bin/env python
__doc__ = """

Inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import ConfigParser
import h5py
import numpy as np
import os
import sys
import time

import config
from DataProvider.python.forward import ForwardScanner
from DataProvider.python.transform import flip, revert_flip

def run(gpu, cfg_path, aug_spec):
    """
    Run inference loop.
    """
    # Initialize.
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    # Forward config.
    cfg = config.ForwardConfig(cfg_path)

    # Create an inference net.
    net = cfg.net()

    # Create net spec.
    net_spec = dict()
    for i in net.inputs:
        net_spec[i] = net.blobs[i].data.shape[-3:]

    # Create data provider.
    dp = cfg.get_data_provider(net_spec)

    # Scan params.
    scan_params = eval(cfg.get('forward','scan_params'))
    flip_range  = eval(cfg.get('forward','flip_range'))
    scan_list   = eval(cfg.get('forward','scan_list'))

    # Create scan spec.
    scan_spec = dict()
    for i in scan_list:
        scan_spec[i] = net.blobs[i].data.shape[-4:]

    # Save path.
    save_prefix = cfg.get('forward','save_prefix')
    save_path = os.path.dirname(save_prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Fix flip range.
    flip_range = aug_spec.keys()

    # Forward scan.
    for dataset in dp.datasets:
        # Inference on each dataset.
        did = dataset.params['dataset_id']
        print 'Forward scan dataset {}'.format(did)

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
            if aug_id in aug_spec:
                prefix  = os.path.basename(save_prefix)
                dirname = save_path + '/' + aug_spec[aug_id]
                if not os.path.exists(dirname): os.makedirs(dirname)
                for key in accum.outputs.data.iterkeys():
                    fname = '{}/{}_dataset{}_{}.h5'.format(dirname, prefix, did+1, key)
                    if os.path.exists(fname): os.remove(fname)
                    print 'Save {}...'.format(fname)
                    f = h5py.File(fname)
                    output = accum.outputs.get_data(key)
                    f.create_dataset('/main', data=output/count)
                    f.close()


if __name__ == '__main__':

    import argparse

    dsc = 'Inference w/ test-time augmentation.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('gpu', type=int, help='gpu device id.')
    parser.add_argument('cfg', help='inference config.')
    parser.add_argument('-aug_spec', help='augmentation spec.')

    args = parser.parse_args()
    run(args.gpu, args.cfg, eval(args.aug_spec))
