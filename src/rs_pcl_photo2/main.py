from tarfile import FIFOTYPE
from lib.depth_image import image_to_array_2d, calc_normals_from_depth
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import fftpack
from td_fft.load_img import *
from td_fft.fourier_mask import *

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def fourier_iterator(image, value_list):
    for i in value_list:
        plot1 = fourier_masker_ver(image, i)
        plot2 = fourier_masker_hor(image, i)

        plot1.savefig("fft_vertical_image" + str(i) + ".png" )
        plot2.savefig("fft_horizontal_image" + str(i) + ".png" )

if __name__ == "__main__":
    from lib.util import mkVisual
    import sys
    from functools import partial

    mkVisual = partial(mkVisual, colormap=cv.COLORMAP_CIVIDIS)
    fname = sys.argv[1] if len(sys.argv) > 1 else "depth_images/rs-photo-00-depth.png"
    dimg = image_to_array_2d(cv.imread(fname, cv.IMREAD_UNCHANGED))
    print(dimg.ndim)
    # dark_image_grey_fourier = dark_image_grey_fourier(dimg)
    fourier_iterator(dimg, [0.5,3,30])
    # plt.show()