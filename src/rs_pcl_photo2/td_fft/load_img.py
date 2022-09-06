import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


def dark_image_grey(dark_image):
    img = rgb2gray(dark_image)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(img, cmap='gray');

def dark_image_grey_fourier(dark_image_grey):
    img  = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(img)), cmap='gray');


def fourier_masker_ver(image, i):
    f_size = 15
    if(image.ndim==3):
        image = rgb2gray(image)
    else:
        image = image[:]

    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    dark_image_grey_fourier[:225, 235:240] = i
    dark_image_grey_fourier[-225:,235:240] = i
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size);
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', fontsize = f_size);