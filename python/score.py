#!/usr/bin/env python
__doc__ = """

Test loop.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
from collections import OrderedDict
import numpy as np
import time

def test_net(iter, solver, test_iter, sampler, keys, monitor=None):
    """
    Run test loop.
    """

    # Test net.
    # solver.test_nets[0].share_with(solver.net)
    # net = solver.test_nets[0]
    net = solver.net

    # Monitoring.
    loss = OrderedDict()
    nmsk = OrderedDict()
    for k in keys:
        loss[k] = 0.0
        nmsk[k] = 0.0

    # Timing.
    start = time.time()
    total_time = 0.0

    # Test loop.
    for i in range(1,test_iter+1):

        # Set inputs.
        sample = sampler(imgs=['input'])
        for k, v in sample.iteritems():
            if k in net.blobs:
                # Assume a sole example in minibatch (single output patch).
                shape = (1,) + v.shape
                net.blobs[k].reshape(*shape)
                net.blobs[k].data[0,...] = v

        # Run forward pass.
        net.forward()

        # Update stats.
        for k in loss.iterkeys():
            loss[k] += net.blobs[k+'_loss'].data
            if k+'_mask' in net.blobs:
                nmsk[k] += np.count_nonzero(net.blobs[k+'_mask'].data>0)
            else:
                nmsk[k] += net.blobs[k+'_loss'].data.size

        # Elapsed time.
        total_time += time.time() - start
        start = time.time()

    # Normalize.
    elapsed = total_time/test_iter
    for k in loss.iterkeys():
        loss[k] /= nmsk[k]
    # Bookkeeping.
    if monitor is not None:
        monitor.append_test(iter, loss)
    # Display.
    disp = '[test] Iteration %d, ' % iter
    for k, v in loss.iteritems():
        disp += '%s: %.3f, ' % (k, v)
    disp += 'elapsed: %.3f s/iter.'  % elapsed
    print disp
