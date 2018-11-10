#!/usr/bin/python3

import sys
import os
import numpy
import random
import argparse

from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

from scipy import ndarray
from scipy import misc

from io import BytesIO
from PIL import Image, ImageDraw

from util import read_image

class Cell(object):
    def __init__(self, index, start_w, start_h, len_w, len_h, data):
        self.index = index
        self.start_w = start_w
        self.start_h = start_h
        self.len_w = len_w
        self.len_h = len_h
        self.data = data
        self.chosen = -1

def show_bytes(data):
    img = Image.frombytes("RGB", (data.shape[1], data.shape[0]), data)
    img.show()

def make_hexagon(data):
    im = Image.frombytes("RGB", (data.shape[1], data.shape[0]), data)
    mask = Image.new('RGBA', im.size)
    d = ImageDraw.Draw(mask)
    (sz_w, sz_h) = data.shape[0:2]
    d.polygon(((int(2*sz_h/3)-1, 0), (int(sz_h/3)+1, 0), (0, int(sz_w/2)), (int(sz_h/3)+1, sz_w), (int(2*sz_h/3)-1, sz_w), (sz_h-1, int(sz_w/2))), fill='#000')
    out = Image.new('RGB', im.size)
    out.paste(im, (0, 0), mask)
    return numpy.array(out)

def read_collection(collection_dir):
    images = []
    for x in os.listdir(collection_dir):
        images.append(read_image("{}{}".format(collection_dir, x)))
    return images

def mean_color(img):
    ret = numpy.mean(img, axis=(0, 1))
    if isinstance(ret, numpy.float64):
        # Convert from black and white to RGB.
        return numpy.array([ret, ret, ret])
    return ret

def rect_of_colour(w, h, col):
    return numpy.repeat([numpy.repeat([col], h, axis=0)], w, axis=0).astype('uint8')

def generate(K_TARGET_IMG, K_COLLECTION_DIR, K_HORIZONTAL, K_RANDOM_PLACE, K_GRID_ALLOW_DUPLICATES, K_HEXAGONAL):
    collection = read_collection(K_COLLECTION_DIR)
    if K_HEXAGONAL:
        collection = [make_hexagon(x) for x in collection]

    target = read_image(K_TARGET_IMG)
    horizontal_pieces_count = K_HORIZONTAL
    
    (small_w, small_h) = collection[0].shape[0:2]
    (original_w, original_h) = target.shape[0:2]

    delta_h = small_h
    if K_HEXAGONAL:
        # Skew the distance between consecutive hexagonal pieces
        delta_h = int(2 * small_h / 3)

    magnification = 1.0 * small_w * horizontal_pieces_count / original_w
    vertical_pieces_count = int((magnification * original_h) / delta_h)

    (big_w, big_h) = (horizontal_pieces_count * small_w, vertical_pieces_count * delta_h)
    target = misc.imresize(target, (big_w, big_h))

    print("Generating...")
    
    kd = KDTree([mean_color(x) for x in collection])
    grid = {}

    if K_RANDOM_PLACE:
        for i in range(horizontal_pieces_count * vertical_pieces_count * 5):
            start_w = random.randint(0, big_w - small_w)
            start_h = random.randint(0, big_h - small_h)
            index = (i, i)
            cell_data = target[start_w:start_w + small_w, start_h:start_h + small_h].copy()
            cell = Cell(index, start_w, start_h, small_w, small_h, cell_data)
            grid[index] = cell
    else:
        horizontal_limit = horizontal_pieces_count
        vertical_limit = vertical_pieces_count
        if K_HEXAGONAL:
            horizontal_limit -= 1
            vertical_limit -= 1
        for i in range(horizontal_limit):
            for j in range(vertical_limit):
                start_w = small_w * i
                start_h = delta_h * j
                if K_HEXAGONAL and j % 2 == 1:
                    start_w += int(small_w / 2)
                
                index = (i, j)
                cell_data = target[start_w:start_w + small_w, start_h:start_h + small_h].copy()
                cell = Cell(index, start_w, start_h, small_w, small_h, cell_data)
                grid[index] = cell

    for index in grid:
        avg_color = mean_color(grid[index].data)
        if K_GRID_ALLOW_DUPLICATES:
            grid[index].chosen = kd.query(avg_color, k=1)[1]
        else:
            (i, j) = index
            def get_choice(i, j):
                if (i, j) in grid:
                    return grid[(i, j)].chosen
                return -1
            already_used = []
            already_used.append(get_choice(i - 1, j))
            already_used.append(get_choice(i + 1, j))
            already_used.append(get_choice(i, j - 1))
            already_used.append(get_choice(i, j + 1))
            if K_HEXAGONAL:
                # Hexagons have six neighbors
                already_used.append(get_choice(i - 1, j - 1))
                already_used.append(get_choice(i - 1, j + 1))
            for option in kd.query(avg_color, k=7)[1]:
                if option not in already_used:
                    grid[index].chosen = option
                    break
    
    target.fill(0)

    for index in grid:
        start_w = grid[index].start_w
        start_h = grid[index].start_h
        len_w = grid[index].len_w
        len_h = grid[index].len_h
        if K_RANDOM_PLACE:
            target[start_w:start_w + len_w, start_h:start_h + len_h] = collection[grid[index].chosen]
        else:
            target[start_w:start_w + len_w, start_h:start_h + len_h] += collection[grid[index].chosen]
    
    if K_HEXAGONAL:
        # Crop blind hexagon margins
        target = target[small_w:big_w - small_w, small_h:big_h - small_h].copy()
    
    return target

def main():
    parser = argparse.ArgumentParser(description='Realizarea imaginilor mozaic.')
    parser.add_argument('target', type=str, help='Target picture to recreate.')
    args = parser.parse_args()

    # default values
    K_COLLECTION_DIR = "./data/automobile/"
    K_HORIZONTAL = 30
    K_RANDOM_PLACE = False
    K_GRID_ALLOW_DUPLICATES = False
    K_HEXAGONAL = True

    ld = generate(args.target, K_COLLECTION_DIR, K_HORIZONTAL, K_RANDOM_PLACE, K_GRID_ALLOW_DUPLICATES, K_HEXAGONAL)
    img = Image.frombytes("RGB", (ld.shape[1], ld.shape[0]), ld)
    img.show()

# Usage: $ python3 main.py ./data/imaginiTest/ferrari.jpeg

if __name__ == '__main__':
    sys.exit(main())