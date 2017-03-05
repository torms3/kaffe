#!/usr/bin/env python
__doc__ = """

Meta-inference.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""
import ConfigParser
import os
import sys
import tempfile

def run(cfg_path, sysline):
    # Config file.
    cfg = config.TrainConfig(cfg_path)

    # Iterate through a specified range of model weights.
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
        f = tempfile.NamedTemporaryFile(delete=False, dir=os.getcwd())
        temp = os.path.basename(f.name)
        cfg.write(f)
        f.close()

        # Inference.
        sysline = 'python ' + sysline + ' ' + temp
        os.system(sysline)

        # Delete temporary file.
        if os.path.exists(temp):
            os.remove(temp)


if __name__ == '__main__':

    import argparse

    dsc = 'Meta-inference.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('sys', help='sysline.')
    parser.add_argument('cfg', help='meta config.')

    args = parser.parse_args()
    run(args.cfg, args.sysline)
