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
from utils import crop_center

import config
from DataProvider.python.forward import ForwardScanner
from DataProvider.python.transform import flip, revert_flip

def run(gpu, cfg_path):
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

    # Center-crop.
    crop = cfg.get('forward','crop')
    if crop is not None:
        print "Crop %s" % crop
        crop = eval(crop)

    # Forward scan.
    for dataset in dp.datasets:
        # Inference on each dataset.
        did = dataset.params['dataset_id']
        print 'Forward scan dataset {}'.format(did)

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

        # Save as file.
        for key in fs.outputs.data.iterkeys():
            fname = '{}_dataset{}_{}.h5'.format(save_prefix, did+1, key)
            print 'Save {}...'.format(fname)
            if os.path.exists(fname): os.remove(fname)
            f = h5py.File(fname)
            output = fs.outputs.get_data(key)
            if crop is not None:
                output = crop_center(output, crop)
            f.create_dataset('/main', data=output)
            f.close()


if __name__ == '__main__':

    import argparse

    dsc = 'Inference.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('gpu', type=int, help='gpu device id.')
    parser.add_argument('cfg', help='inference config.')

    args = parser.parse_args()
    run(args.gpu, args.cfg)
