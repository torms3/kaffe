#!/usr/bin/env python
__doc__ = """

Meta-inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""
import ConfigParser
import os
import sys
import tempfile

# Config file.
cfg = ConfigParser.ConfigParser()
cfg.read(sys.argv[3])

wrange = eval(cfg.get('forward','wrange'))
for w in wrange:
    print 'Iteration {}...'.format(w)

    # Iteration-specific setup.
    cfg.set('forward','witer',w)

    # Check save path.
    save_prefix = cfg.get('forward','save_prefix')
    save_path = os.path.dirname(save_prefix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Temporary config file.
    f = tempfile.TemporaryFile()
    cfg.write(f)

    # Inference.
    sysline = 'python {} {} {}'.format(sys.argv[1],sys.argv[2],f.name)
    os.system(sysline)

    # Close temporary file.
    f.close()
