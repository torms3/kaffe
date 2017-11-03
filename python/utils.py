#!/usr/bin/env python
__doc__ = """

Utility functions.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

def crop_center(img, shape):
    sz,sy,sx = shape[-3:]
    oz,oy,ox = [(a - b)//2 for a, b in zip(img.shape[-3:],shape[-3:])]
    return img[...,oz:oz+sz,oy:oy+sy,ox:ox+sx]