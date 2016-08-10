import caffe
import numpy as np
import os
import sys
import time

import parser
import score
import stats

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Create solver.
fname  = sys.argv[2]
config = parser.SolverParser().parse(fname)
solver = caffe.SGDSolver(fname)

# Monitoring.
monitor = stats.LearningMonitor()
stats = dict(loss=0.0, cerr=0.0, nmsk=0.0)

# Load net, if any.
prefix = eval(config.get('solver','snapshot_prefix'))
last_iter = 0
if len(sys.argv) > 3:
    last_iter = int(sys.argv[3])
    # Solver state.
    fname = '{}_iter_{}.solverstate.h5'.format(prefix, last_iter)
    solver.restore(fname)
    # Snapshot.
    # fname = '{}_iter_{}.caffemodel.h5'.format(prefix, last_iter)
    # solver.net.copy_from(fname)
    # Stats.
    fname = '{}_iter_{}.statistics.h5'.format(prefix, last_iter)
    monitor.load(fname)
net = solver.net

print 'Start training...'
print 'Start from ', last_iter + 1

# Timing.
total_time = 0.0
start = time.time()

# Training loop.
max_iter = config.getint('solver','max_iter')
for i in range(last_iter+1,max_iter+1):

    # Set inputs.
    sample = dp.random_sample()
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
    display = config.getint('solver','display')
    if i % display == 0:
        # Normalize.
        elapsed = total_time/display
        stats['loss'] /= stats['nmsk']
        stats['cerr'] /= stats['nmsk']
        # Bookkeeping.
        monitor.append_train(i, stats)
        # Display.
        base_lr = config.get('solver','base_lr')
        print 'Iteration %7d, loss: %.3f, cerr: %.3f,'      \
              'learning rate: %.6f, elapsed: %.3f s/iter'   \
                % (i, stats['loss'], stats['cerr'], base_lr, elapsed)
        # Reset.
        for key in stats.iterkeys():
            stats[key] = 0.0
        total_time = 0.0
        start = time.time()

    # Test loop.
    test_interval = 500
    if i % test_interval == 0:
        score.test_net(i, solver, config, monitor)
        start = time.time()  # Skip test time.

    # Save stats.
    snapshot = config.getint('solver','snapshot')
    if i % snapshot == 0:
        fname = '{}_iter_{}.statistics.h5'.format(prefix, i)
        monitor.save(fname, elapsed)
