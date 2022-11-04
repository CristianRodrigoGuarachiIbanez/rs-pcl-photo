#!/bin/env python3
from lib.depth_image import image_to_array_2d, calc_normals_from_depth
from td_fft.image_reconstruction import calculate_2dft, calculate_2dift
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack


class FFT:
    def __init__(self, file=None):
        self._image = self._read_image(file)
        self._filter = None

    def getImage(self):
        return self._image

    def getFilter(self):
        return self._filter

    def _read_image(self, file=None):
        image = cv.imread(file, cv.IMREAD_UNCHANGED)
        if image.size == 0:
            return None
        return image_to_array_2d(image)

    def set_filter(self, file=None, method=1):
        if file:
            return cv.imread(file, cv.IMREAD_GRAYSCALE)
        else:
            if self._image.size != 0 and method == 1:
                return calculate_2dft(self._image)
            elif self._image.size != 0 and method == 2:
                return self._calculate_filter(self._image)
            else:
                raise Exception("No image was loaded!")

    @staticmethod
    def _calculate_filter(dimg):
        """
        https://wsthub.medium.com/python-computer-vision-tutorials-image-fourier-transform-part-3-e65d10be4492
        """
        return np.fft.fftshift(np.fft.fft2(dimg))

    @staticmethod
    def _recalculate_image(fft):
        return np.fft.ifft2(np.fft.ifftshift(fft))

    @staticmethod
    def _set_filter_frequencies(copy, keep_fraction):

        # Set r and c to be the number of rows and columns of the array.
        r, c = copy.shape

        # Set to zero all rows with indices between r*keep_fraction and r*(1-keep_fraction):
        copy[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0

        # Similarly with the columns:
        copy[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0

        return copy
    def calculate_normals(self):
        if self._image.size == 0:
            self._normals = calc_normals_from_depth(self._image)
        else:
            raise Exception("No image was loaded!")

    def calculate_fft(self, image, filter, method):
        assert image.size > 0, "no image has been loaded"

        # assert self._image.shape
        if filter is None:
            if method == 1:
                return self._calculate_filter(image)
            else:
                return calculate_2dft(image)
            # Call ff a copy of the original transform. Numpy arrays have a copy method for this purpose.
        else:
            assert filter.size > 0, "no filter image has been loaded!"
            assert image.shape == filter.shape, "shapes are not equal to vector multiplication! "
            if method == 1:
                return self._calculate_filter(image) * filter
            else:
                return calculate_2dft(image) * filter

    def calculate_mask(self, file, filter, mask, method):
        self._filter = self.set_filter(file, method)
        if type(filter) is not np.ndarray:
            if mask:
                return self.calculate_fft(self._image, self._filter, method)
            else:
                return self.calculate_fft(self._image, None, method)
        else:
            return self.calculate_fft(self._image, filter, method)

    def reconstruct_image(self, file=None, filter=None, mask=None, method=1, freq=0.499211):
        mask = self.calculate_mask(file, filter, mask, method)
        if freq:
            mask = self._set_filter_frequencies(mask.copy(), freq)
        if method == 1:
            return self._recalculate_image(mask)
        else:
            return calculate_2dift(mask)

    def recover_image(self, img: str, fname="image.png"):
        if img == "original":
            plt.imshow(self._image)
            plt.savefig(fname)
        elif img == "mask":
            plt.imshow(np.log(abs(self._filter)))
            plt.savefig(fname)
        elif img == "reco":
            self._recon = self.reconstruct_image()
            assert self._recon.size > 0, "no recontructed image could be calculated!"
            plt.imshow(abs(self._recon), plt.cm.gray)
            plt.savefig(fname)
        else:
            print("[Error]: the argument {} is not supported".format(img))
            print("use one of the following arguments: original, mask, reco")

def show( dimg=None, method=None, fname="image.png"):
    if method == "log":
        plt.imshow(np.log(abs(dimg)))
    elif method == "gray":
        plt.imshow(abs(dimg), plt.cm.gray)
    else:
        plt.imshow(abs(dimg))
    plt.savefig(fname)

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()


if __name__ == "__main__":
    pass

