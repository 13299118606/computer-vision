#!/usr/bin/python3

import math
import skimage
import sklearn.preprocessing
import numpy
from PIL import Image
from skimage.util import view_as_windows as viewW

def read_image(file_name):
    return normalize_image(skimage.io.imread(file_name))

def normalize_image(img):
    ret = skimage.img_as_ubyte(skimage.transform.resize(img, img.shape, mode='constant', anti_aliasing=True))
    if len(ret.shape) == 2:
        # Transform B&W to RGB.
        (height, width) = ret.shape
        ret = numpy.stack((ret,) * 3, axis=-1)
    if len(ret.shape) == 3 and ret.shape[2] == 4:
        # Transform RGBA to RGB.
        ret = numpy.delete(ret, axis=2, obj=3)
    return ret

def show_bytes(data, path=None):    
    data = data.astype(numpy.float64)
    data = data.astype(numpy.uint8)
    img_type = 'RGB'
    if len(data.shape) == 2:
        img_type = 'L' # grayscale image
    img = Image.frombytes(img_type, (data.shape[1], data.shape[0]), data)
    if path == None:
        img.show()
    else:
        img.save(path)

def normc(a):
    return sklearn.preprocessing.normalize(a, axis=0, norm='l2')

def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    result = numpy.empty((block_size[0] * block_size[1], sx * sy))
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result

def col2im(mtx, image_size, block_size):
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = numpy.zeros(image_size)
    weight = numpy.zeros(image_size)
    col = 0
    for i in range(sy):
        for j in range(sx):
            result[j:j + p, i:i + q] += mtx[:, col].reshape(block_size, order='F')
            weight[j:j + p, i:i + q] += numpy.ones(block_size)
            col += 1
    return result / weight

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

from skimage.measure import compare_ssim as ssim

numpy.warnings.filterwarnings('ignore')

p = 12;                 # patch size
s = 5;                  # sparsity
N = 200;                # total number of patches
n = 256;                # dictionary size
K = 50;                 # DL iterations
sigma = 7;              # noise standard deviation

# Add noise to original image and vectorize
I = read_image("easy.png")
show_bytes(I, "s_denoise_sliding_I.png")
I = I[:, :, 1].copy().astype('float64')
Inoisy = I + sigma * numpy.random.randn(I.shape[0], I.shape[1])
show_bytes(Inoisy,  "s_denoise_sliding_Inoisy.png")

print("ssim none: ", ssim(I, I, data_range=I.max() - I.min()))
print("psnr none: ", psnr(I, I))

print("ssim noisy: ", ssim(I, Inoisy, data_range=Inoisy.max() - Inoisy.min()))
print("psnr noisy: ", psnr(I, Inoisy))

# Extract distinct patches and center
Ynoisy = im2col(Inoisy, (p, p))
Ymean = Ynoisy.mean(axis=0)
Ynoisy = Ynoisy - numpy.tile(Ymean, [Ynoisy.shape[0], 1])

# Select sample patches for training
ch = numpy.random.permutation(Ynoisy.shape[1])[:N]
Y = Ynoisy[:,ch].T
print(Y.shape)

# Training dictionary
from sklearn.decomposition import DictionaryLearning
dico = DictionaryLearning(n, transform_algorithm='omp', alpha=s, random_state=0, verbose=False)
dico.fit(Y)

# Testing the validity of the sparse representation
Xt = dico.transform(Y)
print(Xt.shape)
numpy.testing.assert_array_almost_equal(numpy.dot(Xt, dico.components_), Y, decimal=1)

# Generating sparse representation for entire image
Xc = dico.transform(Ynoisy.T)
print(Xc.T.shape)
# D * X
A = numpy.dot(Xc, dico.components_).T

# Inverse centering, image restoration and output
A = A + numpy.tile(Ymean, [Ynoisy.shape[0], 1])
Ic = col2im(A, (I.shape[0], I.shape[1]), (p, p))
show_bytes(Ic, "s_denoise_sliding_Ic.png")

# Statistics
print("ssim final: ", ssim(I, Ic, data_range=Ic.max() - Ic.min()))
print("psnr final: ", psnr(I, Ic))