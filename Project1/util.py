#!/usr/bin/python3

import skimage.io
import numpy
from scipy import misc

def read_image(file_name):
    return normalize_image(skimage.io.imread(file_name))

def normalize_image(img):
    ret = misc.imresize(img, 100)
    if len(ret.shape) == 2:
        # Transform B&W to RGB.
        (height, width) = ret.shape
        ret = numpy.stack((ret,) * 3, axis=-1)
    if len(ret.shape) == 3 and ret.shape[2] == 4:
        # Transform RGBA to RGB.
        ret = numpy.delete(ret, axis=2, obj=3)
    return ret