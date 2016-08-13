#!/usr/bin/env python
__doc__ = """

TrainConfig.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import ConfigParser
import os

class TrainConfig(ConfigParser.ConfigParser):
    """
    TrainConfig.
    """

    def __init__(self, fname):
        """Initialize TrainConfig."""
        ConfigParser.ConfigParser.__init__(self)
        self.read(fname)

    def get_solver(self):
        """Create a temporary solver file and get solver from it."""
        # Create a temporary solver file.
        fname = '__solver__.prototxt'
        f = open(fname, 'w')
        f.write(self.to_proto())
        f.close()
        # Get solver from file.
        solver = caffe.get_solver_from_file(fname)
        # Remove the temporary solver file and return solver.
        os.remove(fname)
        return solver

    def to_proto(self):
        """Convert [solver] section to prototxt."""
        prototxt = str()
        opts = self.options('solver')
        for opt in opts:
            val = self.get('solver',opt)
            prototxt += opt + ': ' + val + '\n'
        return prototxt


if __name__ == "__main__":

    config_path = 'train.cfg.example'
    train_cfg = TrainConfig(config_path)
    print train_cfg.to_proto()

    solver = train_cfg.get_solver()
    print solver.max_iter
