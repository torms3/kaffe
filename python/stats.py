#!/usr/bin/env python
__doc__ = """

Managing training statistics.

Jingpeng Wu <jingpeng.wu@gmail.com>,
Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""
import numpy as np
import h5py
import os
import shutil


class LearningMonitor:
    """
    LearningMonitor.
    """

    def __init__(self, fname=None):
        """Initialize LearningMonitor."""
        if fname is None:
            self.train = dict()  # Train stats.
            self.test  = dict()  # Test stats.
        else:
            self.load(fname)

    def append_train(self, iter, data):
        """Add train stats."""
        self._append(iter, data, 'train')

    def append_test(self, iter, data):
        """Add test stats."""
        self._append(iter, data, 'test')

    def get_last_iter(self):
        """Return the last iteration number."""
        ret = 0
        if 'iter' in self.train and 'iter' in self.test:
            ret = max(self.train['iter'][-1],self.test['iter'][-1])
        return ret

    def load(self, fname):
        """Initialize by loading from a h5 file."""
        assert(os.path.exists(fname))
        f = h5py.File(fname, 'r', driver='core')
        # Train stats.
        train = f['/train']
        for key, data in train.iteritems():
            self.train[key] = list(data.value)
        # Test stats.
        test = f['/test']
        for key, data in test.iteritems():
            self.test[key] = list(data.value)
        f.close()

    def save(self, fname, elapsed, base_lr=0):
        """Save stats."""
        if os.path.exists(fname):
            os.remove(fname)
        # Crate h5 file to save.
        f = h5py.File(fname)
        # Train stats.
        for key, data in self.train.iteritems():
            f.create_dataset('/train/{}'.format(key), data=data)
        # Test stats.
        for key, data in self.test.iteritems():
            f.create_dataset('/test/{}'.format(key), data=data)
        # Iteration speed in (s/iteration).
        f.create_dataset('/elapsed', data=elapsed)
        f.create_dataset('/base_lr', data=base_lr)
        f.close()

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _append(self, iter, data, phase):
        assert phase=='train' or phase=='test'
        d = getattr(self, phase)
        # Iteration.
        if 'iter' not in d:
            d['iter'] = list()
        d['iter'].append(iter)
        # Stats.
        for key, val in data.iteritems():
            if key not in d:
                d[key] = list()
            d[key].append(val)
