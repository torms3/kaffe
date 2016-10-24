#!/usr/bin/env python
__doc__ = """

Meta-inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""
import ConfigParser
import os
import sys

# Config file.
cfg = ConfigParser.ConfigParser()
cfg.read(sys.argv[3])

# Temporary config file name.
temp = 'temp.cfg'

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
    f = open(temp,'w')
    cfg.write(f)
    f.close()

    # Inference.
    sysline = 'python {} {} {}'.format(sys.argv[1],sys.argv[2],temp)
    os.system(sysline)

    # Delete temporary file.
    os.remove(temp)
