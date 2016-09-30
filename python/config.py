#!/usr/bin/env python
__doc__ = """

TrainConfig.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import caffe
import ConfigParser
import os

from DataProvider.python.data_provider import VolumeDataProvider

class Config(ConfigParser.ConfigParser):
    """
    Config interface.
    """

    def __init__(self, fname):
        """Initialize Config."""
        ConfigParser.ConfigParser.__init__(self)
        self.read(fname)

    def get_data_provider(self, net_spec):
        raise NotImplementedError


class TrainConfig(Config):
    """
    TrainConfig.
    """

    def __init__(self, fname):
        """Initialize TrainConfig."""
        Config.__init__(self, fname)

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

    def get_data_provider(self, net_spec):
        """Create train & test data providers."""
        dp = dict()
        # Data spec path.
        dspec_path = self.get('train','dspec_path')
        # Create train data provider.
        params = self._get_data_provider_params('train')
        dp['train'] = VolumeDataProvider(dspec_path, net_spec, params)
        # Create test data provider.
        params = self._get_data_provider_params('test')
        dp['test'] = VolumeDataProvider(dspec_path, net_spec, params)
        return dp

    def _get_data_provider_params(self, phase):
        """Create a parameter dictionary for data provider."""
        assert phase in ['train','test']
        params = dict()
        params['drange'] = eval(self.get(phase,'drange'))
        if self.has_option(phase,'dprior'):  # Optional.
            params['dprior'] = eval(self.get(phase,'dprior'))
        if self.has_option(phase,'border'):  # Optional.
            params['border'] = eval(self.get(phase,'border'))
        if self.has_option(phase,'augment'):  # Optional.
            params['augment'] = eval(self.get(phase,'augment'))
        return params


class ForwardConfig(Config):
    """
    Config for inference.
    """

    def __init__(self, fname):
        """Initialize ForwardConfig."""
        Config.__init__(self, fname)

    def net(self):
        """Create an inference net."""
        model   = self.get('forward','model')
        weights = self.get('forward','weights')
        return caffe.Net(model, weights, caffe.TEST)

    def get_data_provider(self, net_spec):
        """Create a data provider for inference."""
        # Data spec path.
        dspec_path = self.get('forward','dspec_path')
        # Params for data provider.
        params = dict()
        params['drange'] = eval(self.get('forward','drange'))
        params['border'] = eval(self.get('forward','border'))
        # Create data provider.
        return VolumeDataProvider(dspec_path, net_spec, params)


if __name__ == "__main__":

    config_path = 'train.cfg.example'
    train_cfg = TrainConfig(config_path)
    print train_cfg.to_proto()

    solver = train_cfg.get_solver()
    print solver.max_iter
