#!/bin/env python3

from tarfile import FIFOTYPE
from lib.depth_image import image_to_array_2d, calc_normals_from_depth
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import fftpack
from .2d_fft.load_img import *

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

if __name__ == "__main__":
    from lib.util import mkVisual
    import sys
    from functools import partial

    mkVisual = partial(mkVisual, colormap=cv.COLORMAP_CIVIDIS)
    fname = sys.argv[1] if len(sys.argv) > 1 else "depth_images/rs-photo-00-depth.png"
    dimg = image_to_array_2d(cv.imread(fname, cv.IMREAD_UNCHANGED))

    normals = calc_normals_from_depth(dimg)
    print(normals.dtype, normals.shape)
    # cv.imshow("normals", mkVisual(normals))
    # while ord(cv.waitKey(0) not in "qQ"):
       # pass
    im_fft = fftpack.fft2(dimg)

 
    plt.figure()
    plot_spectrum(im_fft)
    plt.title('Fourier transform')


    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.2

    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()

    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

    # Similarly with the columns:
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

    plt.figure()
    plot_spectrum(im_fft2)
    plt.title('Filtered Spectrum')


    # real part for display.
    im_new = fftpack.ifft2(im_fft2).real

    plt.figure()
    plt.imshow(im_new, plt.cm.gray)
    plt.title('Reconstructed Image')
    plt.show()