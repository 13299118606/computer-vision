#!/usr/bin/python3

import sys
import os
import numpy
from main import show_bytes
from PIL import Image

CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictr = pickle.load(fo, encoding='bytes')
    return dictr

def main():

    d = unpickle('./cifar-10-batches-py/data_batch_1')

    counter = {}
    for cat in CIFAR_LABELS:
        directory = './data/{}'.format(cat)
        if not os.path.exists(directory):
            os.mkdir(directory)
        counter[cat] = 0
    
    for i in range(len(d[b'data'])):
        entry = d[b'data'][i]
        label = d[b'labels'][i]
        cat = CIFAR_LABELS[label]
        counter[cat] += 1
        filename = './data/{}/{}.png'.format(cat, counter[cat])
        im = numpy.array(entry).reshape((3, 32, 32))
        im = numpy.stack(im, axis=2)
        img = Image.frombytes("RGB", (im.shape[1], im.shape[0]), im)
        img.save(filename)

if __name__ == '__main__':
    sys.exit(main())