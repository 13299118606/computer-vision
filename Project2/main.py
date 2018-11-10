import sys
import os
import numpy
import random
import argparse
from scipy import ndarray
from scipy import misc
from scipy import ndimage
from io import BytesIO
from PIL import Image, ImageDraw
from util import read_image

def show_bytes(data):
    img_type = 'RGB'
    if len(data.shape) == 2:
        img_type = 'L' # grayscale image
    img = Image.frombytes(img_type, (data.shape[1], data.shape[0]), data)
    img.show()

def rgb2gray(data):
    return numpy.dot(data[..., :3], [0.299, 0.587, 0.114]).astype('uint8')

def sobel_filter(data):
    im = data.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = numpy.hypot(dx, dy)  # magnitude
    mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    return mag.astype('uint8')

def calculeazaEnergie(data):
    return sobel_filter(rgb2gray(data))

def selecteazaDrumVertical(energy, K_metodaSelectareDrum):
    if K_metodaSelectareDrum == 'aleator':
        (n_h, n_w) = energy.shape[0:2]
        x = random.randint(0, n_w - 1)
        d = []
        d.append([0, x])
        for i in range(1, n_h):
            optiune = 0
            if d[i-1][1] == 0:
                optiune = random.randint(0, 1)
            elif d[i-1][1] == n_w - 1:
                optiune = random.randint(-1, 0)
            else:
                optiune = random.randint(-1, 1)
            x = d[i-1][1] + optiune
            d.append([i, x])
        return d
    elif K_metodaSelectareDrum == 'greedy':
        (n_h, n_w) = energy.shape[0:2]
        x = 0
        for j in range(n_w):
            if energy[0][j] < energy[0][x]:
                x = j
        d = []
        d.append([0, x])
        for i in range(1, n_h):
            y = x
            if x < n_w - 1 and energy[i][x + 1] < energy[i][y]:
                y = x + 1
            if x > 0 and energy[i][x - 1] < energy[i][y]:
                y = x - 1
            x = y
            d.append([i, x])
        return d
    elif K_metodaSelectareDrum == 'programareDinamica':
        (n_h, n_w) = energy.shape[0:2]
        pd = energy.copy().astype('int32')
        choice = pd.copy()
        for i in range(1, n_h):
            for j in range(0, n_w):
                y = 0
                if j < n_w - 1 and pd[i - 1][j + 1] < pd[i][j + y]:
                    y = 1
                if j > 0 and pd[i - 1][j - 1] < pd[i][j + y]:
                    y = -1
                pd[i][j] = pd[i - 1][j + y] + energy[i][j]
                choice[i][j] = y
        x = 0
        for j in range(n_w):
            if pd[n_h - 1][j] < pd[n_h - 1][x]:
                x = j
        d = []
        d.append([n_h - 1, x])
        for i in reversed(range(0, n_h - 1)):
            x += choice[i + 1][x]
            d.append([i, x])
        return d
    else:
        raise Exception("Not implemented")

def ploteazaDrumVertical(img, d):
    col = numpy.array([255, 0, 0])
    for pnt in d:
        img[pnt[0], pnt[1], :] = col
    return img

def eliminaDrumVertical(img, d):
    newshape = [x for x in img.shape]
    newshape[1] -= 1
    reduced = numpy.zeros(newshape).astype('uint8')
    for pnt in d:
        reduced[pnt[0], :pnt[1]] = img[pnt[0], :pnt[1]]
        reduced[pnt[0], pnt[1]:] = img[pnt[0], (pnt[1] + 1):]
    return reduced

def dubleazaDrumVertical(img, d):
    newshape = [x for x in img.shape]
    newshape[1] += 1
    reduced = numpy.zeros(newshape).astype('uint8')
    for pnt in d:
        reduced[pnt[0], :pnt[1]] = img[pnt[0], :pnt[1]]
        reduced[pnt[0], (pnt[1] + 1):] = img[pnt[0], pnt[1]:]
        reduced[pnt[0], pnt[1]] = img[pnt[0], pnt[1]]
    return reduced

def leechEnergy(energy, d):
    for pnt in d:
        energy[pnt[0], pnt[1]] = 255
    return energy

def micsoreazaLatimeUnit(target, K_metodaSelectareDrum):
    energy = calculeazaEnergie(target)
    d = selecteazaDrumVertical(energy, K_metodaSelectareDrum)
    return eliminaDrumVertical(target, d)

def micsoreazaLatime(target, K_metodaSelectareDrum, K_pixels):
    for i in range(K_pixels):
        print(i)
        target = micsoreazaLatimeUnit(target, K_metodaSelectareDrum)
    return target

def maresteLatime(target, K_metodaSelectareDrum, K_pixels):
    energy = calculeazaEnergie(target)
    for i in range(K_pixels):
        print(i)
        d = selecteazaDrumVertical(energy, K_metodaSelectareDrum)
        energy = leechEnergy(energy, d)
        energy = dubleazaDrumVertical(energy, d)
        target = dubleazaDrumVertical(target, d)
    return target

def miscoreazaInaltime(target, K_metodaSelectareDrum, K_pixels):
    target = numpy.rot90(target, 1).copy()
    target = micsoreazaLatime(target, K_metodaSelectareDrum, K_pixels)
    target = numpy.rot90(target, 3).copy()
    return target

def maresteInaltime(target, K_metodaSelectareDrum, K_pixels):
    target = numpy.rot90(target, 1).copy()
    target = maresteLatime(target, K_metodaSelectareDrum, K_pixels)
    target = numpy.rot90(target, 3).copy()
    return target

def amplificaContinut(target, K_metodaSelectareDrum, invfactor):
    (original_w, original_h) = target.shape[0:2]
    pixels = int(original_w / invfactor)
    print(pixels)
    target = misc.imresize(target, (original_w + pixels, original_h + pixels))
    target = miscoreazaInaltime(target, K_metodaSelectareDrum, pixels)
    target = micsoreazaLatime(target, K_metodaSelectareDrum, pixels)
    return target

def eliminaObiect(target, rect):
    energy = calculeazaEnergie(target)
    energy[rect[0]:rect[2], rect[1]:rect[3]].fill(0)
    latime = rect[3] - rect[1] + 5
    print(latime)
    for i in range(latime):
        print(i)
        d = selecteazaDrumVertical(energy, 'programareDinamica')
        energy = eliminaDrumVertical(energy, d)
        target = eliminaDrumVertical(target, d)
    return target

def generate(K_targetImg, K_optiuneRedimensionare, K_metodaSelectareDrum, K_pixels):
    target = read_image(K_targetImg)

    if K_optiuneRedimensionare == 'micsoreazaLatime':
        a = target.copy()
        (w, h) = target.shape[0:2]
        r = misc.imresize(target, (w, h - K_pixels))
        o = micsoreazaLatime(target, K_metodaSelectareDrum, K_pixels)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'miscoreazaInaltime':
        a = target.copy()
        (w, h) = target.shape[0:2]
        r = misc.imresize(target, (w - K_pixels, h))
        o = miscoreazaInaltime(target, K_metodaSelectareDrum, K_pixels)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'maresteLatime':
        a = target.copy()
        (w, h) = target.shape[0:2]
        r = misc.imresize(target, (w, h + K_pixels))
        o = maresteLatime(target, K_metodaSelectareDrum, K_pixels)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'maresteInaltime':
        a = target.copy()
        (w, h) = target.shape[0:2]
        r = misc.imresize(target, (w + K_pixels, h))
        o = maresteInaltime(target, K_metodaSelectareDrum, K_pixels)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'maresteAmbele':
        a = target.copy()
        (w, h) = target.shape[0:2]
        r = misc.imresize(target, (w + K_pixels, h + K_pixels))
        o = maresteLatime(target, K_metodaSelectareDrum, K_pixels)
        o = maresteInaltime(o, K_metodaSelectareDrum, K_pixels)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'amplificaContinut':
        a = target.copy()
        r = amplificaContinut(target, K_metodaSelectareDrum, 5.0)
        o = amplificaContinut(target, K_metodaSelectareDrum, 2.0)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'eliminaObiect':
        rect = K_pixels
        a = target.copy()
        r = target.copy()
        r[rect[0]:rect[2], rect[1]:rect[3]].fill(0)
        o = eliminaObiect(target, rect)
        return (a, r, o)
    elif K_optiuneRedimensionare == 'energie':
        return calculeazaEnergie(target)
    pass

def main():
    parser = argparse.ArgumentParser(description='Seam Carving for Content-Aware Image Resizing')
    parser.add_argument('target', type=str, help='Target picture to resize')
    args = parser.parse_args()

    # Options: 'micsoreazaLatime' 'miscoreazaInaltime' 'maresteLatime' 'maresteInaltime' 'maresteAmbele' 'amplificaContinut' 'eliminaObiect'
    K_optiuneRedimensionare = 'micsoreazaLatime'
    # Options: 'aleator' 'greedy' 'programareDinamica'
    K_metodaSelectareDrum = 'programareDinamica'
    K_pixels = 20
    
    ld = generate(args.target, K_optiuneRedimensionare, K_metodaSelectareDrum, K_pixels)
    #show_bytes(ld)

# Usage: $ python3 main.py ./data/castel.jpg

if __name__ == '__main__':
    sys.exit(main())