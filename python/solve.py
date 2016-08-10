import caffe
import ConfigParser
import numpy as np
import os
import sys
import time

import parser
import score
import stats

from DataProvider.python.data_provider import VolumeDataProvider

# Initialize.
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Train config.
train_cfg = ConfigParser.ConfigParser()
train_cfg.read(sys.argv[2])

# Create solver.
fname  = train_cfg.get('train','solver')
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

# Data provider.
dp = dict()
# Common params.
dspec_path = train_cfg.get('train','dspec_path')
net_spec = eval(train_cfg.get('train','net_spec'))
# Create train data provider.
params = dict()
params['border']  = eval(train_cfg.get('train','border_func'))
params['augment'] = eval(train_cfg.get('train','data_augment'))
params['drange']  = eval(train_cfg.get('train','train_range'))
dp['train'] = VolumeDataProvider(dspec_path, net_spec, params)
# Create test data provider.
params = dict()
params['drange'] = eval(train_cfg.get('train','test_range'))
dp['test'] = VolumeDataProvider(dspec_path, net_spec, params)

# Test & test loop parms.
display       = config.getint('solver','display')
test_iter     = train_cfg.getint('train','test_iter')
test_interval = train_cfg.getint('train','test_interval')

print 'Start training...'
print 'Start from ', last_iter + 1

# Timing.
total_time = 0.0
start = time.time()

# Training loop.
max_iter = config.getint('solver','max_iter')
for i in range(last_iter+1,max_iter+1):

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
        base_lr = config.getfloat('solver','base_lr')
        print 'Iteration %7d, loss: %.3f, cerr: %.3f,'      \
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
    snapshot = config.getint('solver','snapshot')
    if i % snapshot == 0:
        fname = '{}_iter_{}.statistics.h5'.format(prefix, i)
        monitor.save(fname, elapsed)
