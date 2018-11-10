#!/usr/bin/python3

import sys
import subprocess
import tempfile
import random
import argparse
from pathlib import Path
from reportlab.pdfgen import canvas
from util import read_image
from scipy import misc
from PIL import Image
from main import generate

def get_pil_image(data):
    img_type = 'RGB'
    if len(data.shape) == 2:
        img_type = 'L' # grayscale image
    return Image.frombytes(img_type, (data.shape[1], data.shape[0]), data)

def limitresize(target, limit):
    (w, h) = target.shape[0:2]
    if w > limit:
        return misc.imresize(target, (limit, int(h/w * limit)))
    return target

def add_picture_to_canvas(c, data, desc):
    c.setFont("Times-Roman", 8)
    img = get_pil_image(data)
    (w, h) = data.shape[0:2]
    c.setPageSize((h + 100, w + 70))
    c.drawInlineImage(img, 50, 10)
    c.drawCentredString(h / 2 + 50, w + 20, desc)
    c.showPage()

# Cerinta a
def cer_a(c):
    (a, r, o) = generate('./data/castel.jpg', 'micsoreazaLatime', 'programareDinamica', 50)
    add_picture_to_canvas(c, a, 'Cerinta (a) Figura 1.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (a) Figura 1.2: Micsorat 50 pixeli in latime cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (a) Figura 1.3: Micsorat 50 pixeli in latime cu content-aware seam carving. (algoritm programareDinamica)')

# Cerinta b
def cer_b(c):
    (a, r, o) = generate('./data/praga.jpg', 'miscoreazaInaltime', 'greedy', 200)
    add_picture_to_canvas(c, a, 'Cerinta (b) Figura 2.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (b) Figura 2.2: Micsorat 100 pixeli in inaltime cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (b) Figura 2.3: Micsorat 100 pixeli in inaltime cu content-aware seam carving. (algoritm greedy)')

# Cerinta c
def cer_c(c):
    (a, r, o) = generate('./data/delfin.jpeg', 'maresteAmbele', 'programareDinamica', 50)
    add_picture_to_canvas(c, a, 'Cerinta (c) Figura 3.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (c) Figura 3.2: Marit 50 pixeli pe ambele directii cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (c) Figura 3.3: Marit 50 pixeli pe ambele directii cu content-aware seam carving. (algoritm programareDinamica)')

# Cerinta d
def cer_d(c):
    (a, r, o) = generate('./data/arcTriumf.jpg', 'amplificaContinut', 'greedy', None)
    add_picture_to_canvas(c, a, 'Cerinta (d) Figura 4.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (d) Figura 4.2: Amplificare continut cu factor 20%. (algoritm greedy)')
    add_picture_to_canvas(c, o, 'Cerinta (d) Figura 4.3: Amplificare continut cu factor 50%. (algoritm greedy)')

# Cerinta e
def cer_e(c):
    (a, r, o) = generate('./data/lac.jpg', 'eliminaObiect', 'programareDinamica', [168, 397, 205, 430])
    add_picture_to_canvas(c, a, 'Cerinta (e) Figura 5.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (e) Figura 5.2: Delimitarea zonei ce trebuie eliminata.')
    add_picture_to_canvas(c, o, 'Cerinta (e) Figura 5.3: Eliminarea unui obiect din imagine. (algoritm programareDinamica)')

# Cerinta f
def cer_f1(c):
    (a, r, o) = generate('./data/capucino.jpeg', 'miscoreazaInaltime', 'greedy', 100)
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 6.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 6.2: Micsorat 100 pixeli in inaltime cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 6.3 (ESEC): Micsorat 100 pixeli in inaltime cu content-aware seam carving. (algoritm greedy)')
    add_picture_to_canvas(c, generate('./data/capucino.jpeg', 'energie', None, None), 'Cerinta (e) Figura 6.4: Partea de sus are noise mai mic decat cea de jos fiind blurata.')

def cer_f2(c):
    (a, r, o) = generate('./data/ship.jpg', 'maresteLatime', 'programareDinamica', 100)
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 7.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 7.2: Marit 100 pixeli pe latime cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 7.3: Marit 100 pixeli pe latime cu content-aware seam carving. (algoritm programareDinamica)')

def cer_f3(c):
    (a, r, o) = generate('./data/stalin.jpg', 'eliminaObiect', 'programareDinamica', [75, 195, 245, 280])
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 8.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 8.2: Delimitarea dusmanului.')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 8.3: Eliminarea.')

def cer_f4(c):
    (a, r, o) = generate('./data/lib.jpg', 'micsoreazaLatime', 'greedy', 100)
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 9.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 9.2: Micsorat 100 pixeli in latime cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 9.3: Micsorat 100 pixeli in latime cu content-aware seam carving. (algoritm greedy)')

def cer_f5(c):
    (a, r, o) = generate('./data/cat.jpg', 'amplificaContinut', 'greedy', None)
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 10.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 10.2: Amplificare continut cu factor 20%. (algoritm greedy)')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 10.3 (ESEC): Amplificare continut cu factor 50%. (algoritm greedy)')

def cer_f6(c):
    (a, r, o) = generate('./data/avion1.jpeg', 'maresteAmbele', 'programareDinamica', 50)
    add_picture_to_canvas(c, a, 'Cerinta (f) Figura 11.1: Imaginea initiala.')
    add_picture_to_canvas(c, r, 'Cerinta (f) Figura 11.2: Marit 50 pixeli pe ambele directii cu imresize.')
    add_picture_to_canvas(c, o, 'Cerinta (f) Figura 11.3: Marit 50 pixeli pe ambele directii cu content-aware seam carving. (algoritm programareDinamica)')


def resutil(name):
    a = read_image(name)
    a = limitresize(a, 300)
    i = get_pil_image(a)
    i.save(name)

def main():
    parser = argparse.ArgumentParser(description='Generate project 1 pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf')
    args = parser.parse_args()
    c = canvas.Canvas(args.output)

    c.setFont("Times-Roman", 17)
    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect Vedere Artificiala #2')
    c.drawCentredString(200, 250, 'Redimensionarea imaginilor cu pastrarea continutului')
    c.drawCentredString(250, 500, 'Ionescu Teodor-Stelian, Grupa 331')
    c.showPage()

    cer_a(c)
    cer_b(c)
    cer_c(c)
    cer_d(c)
    cer_e(c)
    cer_f1(c)
    cer_f2(c)
    cer_f3(c)
    cer_f4(c)
    cer_f5(c)
    cer_f6(c)

    c.save()

# Usage: $ python3 make_pdf.py pdf_name.pdf && open pdf_name.pdf

if __name__ == '__main__':
    sys.exit(main())