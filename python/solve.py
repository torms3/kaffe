#!/usr/bin/env python
__doc__ = """

Training.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import numpy as np
from Queue import Queue
import os
import sys
import threading
import time

import config
import score
import stats as st

def sample_daemon(sampler, f, q):
    while True:
        if not q.full():
            q.put(f(sampler()))
        else:
            q.join()


def run(gpu, cfg_path, async, last_iter=None):
    # Initialize.
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    # Train config.
    cfg = config.TrainConfig(cfg_path)

    # Create solver.
    solver = cfg.get_solver()

    # Monitoring.
    monitor = st.LearningMonitor()
    stats = dict(loss=0.0, nmsk=0.0)

    # Load net, if any.
    prefix = eval(cfg.get('solver','snapshot_prefix'))
    if last_iter is None:
        last_iter = 0
    else:
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
    backend_time = 0.0
    start = time.time()

    # Asynchronous sampler.
    sampler = dp['train']
    f =
    if async:
        q = Queue(maxsize=10)
        t = threading.Thread(target=sample_daemon, args=(sampler, q))
        t.daemon = True
        t.start()

    # Training loop.
    for i in range(last_iter+1, solver.max_iter+1):

        if async:
            sample = q.get(block=True, timeout=None)
            q.task_done()
        else:
            sample = sampler()

        # Set inputs.
        for k, v in sample.iteritems():
            if k in net.blobs:
                # Assume a sole example in minibatch (single output patch).
                shape = (1,) + v.shape
                net.blobs[k].reshape(*shape)
                net.blobs[k].data[0,...] = v

        # Run forward & backward passes.
        backend_start = time.time()
        solver.step(1)
        backend_time += time.time() - backend_start

        # Loss.
        stats['loss'] += net.blobs['loss'].data
        # Number of valid voxels
        stats['nmsk'] += np.count_nonzero(net.blobs['label_mask'].data>0)
        # Elapsed time.
        total_time += time.time() - start
        start = time.time()

        # Display.
        if i % display == 0:
            # Normalize.
            elapsed = total_time/display
            backend = backend_time/display
            stats['loss'] /= stats['nmsk']
            # Bookkeeping.
            monitor.append_train(i, stats)
            # Display.
            base_lr = cfg.getfloat('solver','base_lr')
            print 'Iteration %7d, loss: %.3f, learning rate: %.6f, '     \
                  'backend: %.3f s/iter, elapsed: %.3f s/iter'   \
                    % (i, stats['loss'], base_lr, backend, elapsed)
            # Reset.
            for key in stats.iterkeys():
                stats[key] = 0.0
            total_time = 0.0
            backend_time = 0.0
            start = time.time()

        # Test loop.
        if i % test_interval == 0:
            score.test_net(i, solver, test_iter, dp['test'], monitor)
            start = time.time()  # Skip test time.

        # Save stats.
        if i % snapshot == 0:
            fname = '{}_iter_{}.statistics.h5'.format(prefix, i)
            monitor.save(fname, elapsed)

    if async:
        t.join()

if __name__ == '__main__':

    import argparse

    dsc = 'Train loop.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('gpu', type=int, help='gpu device id.')
    parser.add_argument('cfg', help='train configuration.')
    parser.add_argument('-a','--async', action='store_true')
    parser.add_argument('-iter', type=int, help='resume iteration.')

    args = parser.parse_args()
    run(args.gpu, args.cfg, args.async, args.iter)
