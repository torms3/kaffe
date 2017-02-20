#!/usr/bin/env python
__doc__ = """

Test loop.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import numpy as np
import time

def test_net(iter, solver, test_iter, dp, monitor=None):
    """
    Run test loop.
    """

    # Test net.
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]

    # Monitoring.
    stats = dict(loss=0.0, nmsk=0.0)

    # Timing.
    start = time.time()
    total_time = 0.0

    # Test loop.
    for i in range(1,test_iter+1):

        # Set inputs.
        sample = dp.random_sample()
        for k, v in sample.iteritems():
            # Assume a sole example in minibatch (single output patch).
            shape = (1,) + v.shape
            net.blobs[k].reshape(*shape)
            net.blobs[k].data[0,...] = v

        # Run forward pass.
        net.forward()

        # Loss.
        stats['loss'] += net.blobs['loss'].data
        # Number of valid voxels.
        stats['nmsk'] += np.count_nonzero(net.blobs['label_mask'].data>0)
        # Elapsed time.
        total_time += time.time() - start
        start = time.time()

    # Normalize.
    elapsed = total_time/test_iter
    stats['loss'] /= stats['nmsk']
    # Bookkeeping.
    if monitor is not None:
        monitor.append_test(iter, stats)
    # Display.
    print '[test] Iteration %d, loss: %.3f, elapsed: %.3f s/iter'\
          % (iter, stats['loss'], elapsed)
