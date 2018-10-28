#!/usr/bin/python3

import sys
import subprocess
import tempfile
import random
import argparse

from pathlib import Path
from reportlab.pdfgen import canvas

from util import read_image
from main import generate

from scipy import misc

from PIL import Image

def add_picture_to_canvas(data, desc, c):
    c.setFont("Times-Roman", 8)
    (w, h) = data.shape[0:2]
    img = Image.frombytes("RGB", (h, w), data)
    c.setPageSize((h + 50, w + 70))
    c.drawInlineImage(img, 25, 10)
    c.drawCentredString(h / 2 + 25, w + 20, desc)
    c.showPage()

def imgresize(target):
    LIMIT = 500
    (big_w, big_h) = target.shape[0:2]
    return misc.imresize(target, (LIMIT, int(big_h/big_w * LIMIT)))

def cerinta(imagelist, idx, comment, genfunc, c):
    # sizes = [50, 25, 50, 100, 75, 100]
    sizes = [10, 10, 10, 10, 10, 10]
    num = 0
    for imgpath in imagelist:
        print(imgpath)
        size = sizes[num]
        num += 1
        idx[0] += 1
        label = comment.format(idx[0], imgpath.name.split('.')[0], size)
        img = imgresize(genfunc(imgpath, size))
        add_picture_to_canvas(img, label, c)

def main():
    TARGET_IMGS_DIR = './data/imaginiTest/'
    CIFAR_DIR = './cifar-10-batches-py/'
    COLLECTION_DIR = "./data/colectie/"

    parser = argparse.ArgumentParser(description='Generate project 1 pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf')
    args = parser.parse_args()

    target_path = Path(TARGET_IMGS_DIR)
    c = canvas.Canvas(args.output)

    c.setFont("Times-Roman", 17)
    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect Vedere Artificiala #1')
    c.drawCentredString(200, 250, 'Realizarea imaginilor mozaic')
    c.drawCentredString(250, 500, 'Ionescu Teodor-Stelian, Grupa 331')
    c.showPage()
    
    imagelist = [f for f in target_path.iterdir() if not f.name.startswith('.')]
    img_idx = [0]

    # Cerinta a
    comment = 'Cerinta (a) Figura {}: Mozaic {} din flori dreptunghiulare dispuse Grid, numarPieseMozaicOrizontala={}, criteriul distantei euclidiene dintre culorile medii.'
    genfunc = lambda imgpath, size : generate(imgpath, COLLECTION_DIR, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=True, K_HEXAGONAL=False)
    cerinta(imagelist, img_idx, comment, genfunc, c)

    # Cerinta b
    comment = 'Cerinta (b) Figura {}: Mozaic {} din flori dreptunghiulare dispuse Aleator, numarPieseMozaicOrizontala={}, criteriul distantei euclidiene dintre culorile medii.'
    genfunc = lambda imgpath, size : generate(imgpath, COLLECTION_DIR, K_HORIZONTAL=size, K_RANDOM_PLACE=True, K_GRID_ALLOW_DUPLICATES=True, K_HEXAGONAL=False)
    cerinta(imagelist, img_idx, comment, genfunc, c)

    # Cerinta c
    comment = 'Cerinta (c) Figura {}: Mozaic {} din flori dreptunghiulare dispuse Grid, numarPieseMozaicOrizontala={}, cu proprietatea ca nu exista doua piese adiacente identice.'
    genfunc = lambda imgpath, size : generate(imgpath, COLLECTION_DIR, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=False, K_HEXAGONAL=False)
    cerinta(imagelist, img_idx, comment, genfunc, c)

    # Cerinta d
    genfunc = lambda imgpath, size, collection : generate(imgpath, K_COLLECTION_DIR=collection, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=True, K_HEXAGONAL=False)
    catchoice = ['frog', 'automobile', 'bird', 'cat', 'ship', 'truck']
    comment = 'Cerinta (d) Figura {}: Mozaic {} compus din imagini din setul CIFAR cu eticheta {}'
    num = 0
    for imgpath in imagelist:
        print(imgpath)
        cat = catchoice[num]
        num += 1
        img_idx[0] += 1
        label = comment.format(img_idx[0], imgpath.name.split('.')[0], cat)
        img = imgresize(genfunc(imgpath, 30, './data/{}/'.format(cat)))
        add_picture_to_canvas(img, label, c)
    genfunc = lambda imgpath, size, collection : generate(imgpath, K_COLLECTION_DIR=collection, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=False, K_HEXAGONAL=True)
    catchoice = ['dog', 'deer', 'airplane', 'horse', 'ship', 'cat']
    comment = 'Cerinta (d) Figura {}: Mozaic hexagonal {} compus din imagini din setul CIFAR cu eticheta {}, cu proprietatea ca nu exista doua piese adiacente identice.'
    num = 0
    for imgpath in imagelist:
        print(imgpath)
        cat = catchoice[num]
        num += 1
        img_idx[0] += 1
        label = comment.format(img_idx[0], imgpath.name.split('.')[0], cat)
        img = imgresize(genfunc(imgpath, 30, './data/{}/'.format(cat)))
        add_picture_to_canvas(img, label, c)

    # Cerinta e
    comment = 'Cerinta (e) Figura {}: Mozaic {} din flori hexagonale dispuse Grid, numarPieseMozaicOrizontala={}.'
    genfunc = lambda imgpath, size : generate(imgpath, COLLECTION_DIR, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=True, K_HEXAGONAL=True)
    cerinta(imagelist, img_idx, comment, genfunc, c)

    # Cerinta f
    comment = 'Cerinta (f) Figura {}: Mozaic {} din flori hexagonale dispuse Grid, numarPieseMozaicOrizontala={}, cu proprietatea ca nu exista doua piese adiacente identice.'
    genfunc = lambda imgpath, size : generate(imgpath, COLLECTION_DIR, K_HORIZONTAL=size, K_RANDOM_PLACE=False, K_GRID_ALLOW_DUPLICATES=False, K_HEXAGONAL=True)
    cerinta(imagelist, img_idx, comment, genfunc, c)

    c.save()

# Usage: $ python3 make_pdf.py aici.pdf && open aici.pdf

if __name__ == '__main__':
    sys.exit(main())