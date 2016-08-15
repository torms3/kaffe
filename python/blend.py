#!/usr/bin/env python
__doc__ = """

Functions for blending inference outputs with overlapping window.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np
import math

from box import Box
from tensor import WritableTensorData as WTD
from vector import Vec3d

class Blend(object):
    """
    Blend interface.
    """

    def __init__(self, scanner):
        raise NotImplementedError

    def pull(self):
        raise NotImplementedError


bump_logit = lambda z, y, x, t=1.5: -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)
bump = lambda z, y, x, t, max_logit: math.exp(bump_logit(z,y,x,t) - max_logit)

class BumpBlend(Blend):
    """
    Blending with bump function.
    """

    def __init__(self, scanner):
        """Initialize BumpBlend."""

        assert spec is not None
        assert isinstance(spec, dict)

        # Inference with overlapping window.
        self.max_logits = dict()
        if scanner.mask:
            rmin = scanner.locs[0]
            rmax = scanner.locs[-1]
            for k, v in scanner.scan_spec.iteritems():
                fov = v[-3:]
                a = centered_box(rmin, fov)
                b = centered_box(rmax, fov)
                c = a.merge(b)
                shape = tuple(c.size())
                self.max_logits[k] = WTD(shape, fov=fov, offset=c.min())

        # Compute max_logit for numerical stability.
        if scanner.mask:
            max_logit_window = self._bump_logit_map(v[-3:])
            for k, v in self.max_logits():
                for loc in scanner.locs:
                    v.set_patch(loc, max_logit_window, op='np.max')

    def pull(self):
        pass

    ####################################################################
    ## Private methods.
    ####################################################################

    def _bump_logit_map(self, dim):
        x = range(dim[-1])
        y = range(dim[-2])
        z = range(dim[-3])
        zv, yv, xv = np.meshgrid(z, y, x)
        xv = (xv+1.0)/(dim[-1]+1.0)
        yv = (yv+1.0)/(dim[-2]+1.0)
        zv = (zv+1.0)/(dim[-3]+1.0)
        patch = map(bump_logit, zv, yv, xv)
        return np.assarray(patch, dtype='float32')
