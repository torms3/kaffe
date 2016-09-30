#!/usr/bin/env python
__doc__ = """

Training.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import numpy as np
import os
import sys
import time

import config
import score
import stats

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Train config.
cfg = config.TrainConfig(sys.argv[2])

# Create solver.
solver = cfg.get_solver()

# Monitoring.
monitor = stats.LearningMonitor()
stats = dict(loss=0.0, cerr=0.0, nmsk=0.0)

# Load net, if any.
prefix = eval(cfg.get('solver','snapshot_prefix'))
last_iter = 0
if len(sys.argv) > 3:
    last_iter = int(sys.argv[3])
    # Solver state.
    fname = '{}_iter_{}.solverstate.h5'.format(prefix, last_iter)
    solver.restore(fname)
    # Stats.
    fname = '{}_iter_{}.statistics.h5'.format(prefix, last_iter)
    monitor.load(fname)
net = solver.net

# Create net spec.
net_spec = dict()
for i in net.inputs:
    net_spec[i] = net.blobs[i].data.shape[-3:]

# Create data providers.
dp = cfg.get_data_provider(net_spec)

# Test & test loop parms.
display       = cfg.getint('solver','display')
snapshot      = cfg.getint('solver','snapshot')
test_iter     = cfg.getint('solver','test_iter')
test_interval = cfg.getint('test','interval')

print 'Start training...'
print 'Start from ', last_iter + 1

# Timing.
total_time = 0.0
start = time.time()

# Training loop.
for i in range(last_iter+1, solver.max_iter+1):

    # Set inputs.
    sample = dp['train'].random_sample()
    for k, v in sample.iteritems():
        # Assume a sole example in minibatch (single output patch).
        shape = (1,) + v.shape
        net.blobs[k].reshape(*shape)
        net.blobs[k].data[0,...] = v

    # Run forward & backward passes.
    solver.step(1)

    # Loss.
    stats['loss'] += net.blobs['loss'].data
    # Classification error
    stats['cerr'] += net.blobs['cerr'].data
    # Number of valid voxels
    stats['nmsk'] += np.count_nonzero(net.blobs['label_mask'].data>0)
    # Elapsed time.
    total_time += time.time() - start
    start = time.time()

    # Display.
    if i % display == 0:
        # Normalize.
        elapsed = total_time/display
        stats['loss'] /= stats['nmsk']
        stats['cerr'] /= stats['nmsk']
        # Bookkeeping.
        monitor.append_train(i, stats)
        # Display.
        base_lr = cfg.getfloat('solver','base_lr')
        print 'Iteration %7d, loss: %.3f, cerr: %.3f, '     \
              'learning rate: %.6f, elapsed: %.3f s/iter'   \
                % (i, stats['loss'], stats['cerr'], base_lr, elapsed)
        # Reset.
        for key in stats.iterkeys():
            stats[key] = 0.0
        total_time = 0.0
        start = time.time()

    # Test loop.
    if i % test_interval == 0:
        score.test_net(i, solver, test_iter, dp['test'], monitor)
        start = time.time()  # Skip test time.

    # Save stats.
    if i % snapshot == 0:
        fname = '{}_iter_{}.statistics.h5'.format(prefix, i)
        monitor.save(fname, elapsed)
