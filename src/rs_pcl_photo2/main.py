
# from lib.depth_image import image_to_array_2d
from PyFFT import FFT, show
import matplotlib.pyplot as plt
from td_fft.load_img import fourier_masker_ver
from td_fft.fourier_mask import fourier_masker_hor, detect_frequencies_fft
from filter import gaussianHP, gaussianLP
import numpy as np
# import cv2 as cv
import argparse

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def fourier_iterator(image, value_list):
    for i in value_list:
        plot1 = fourier_masker_ver(image, i)
        plot2 = fourier_masker_hor(image, i)

        plot1.savefig("fft_vertical_image" + str(i) + ".png")
        plot2.savefig("fft_horizontal_image" + str(i) + ".png")

def main(fname, fpath=None, out="reco", pfilter="low", method=1, freq=0.0):

    py_fft = FFT(fname)
    filter = py_fft.set_filter()
    if pfilter == "low":
        mask = gaussianLP(10, filter.shape)
    elif pfilter == "high":
        mask = gaussianHP(10, filter.shape)
    else:
        pass
    recon = py_fft.reconstruct_image(fpath, mask, None, method, freq)

    if out == "filter":
        show(filter, "log", "filter.png")
    elif out == "mask":
        show(mask, "log", "mask.png")
    elif out == "original":
        show(py_fft.getImage(), "last", "original.png")
    elif out == "reco":
        show(recon, "gray", "reco.png")

if __name__ == "__main__":

    # dark_image_grey_fourier = dark_image_grey_fourier(dimg)
    # fourier_iterator(dimg, [-10, 0.5, 0, 10])
    # mean, b = detect_frequencies_fft(image=dimg, thresh=20, vis=True)
    # plt.show()

    parser = argparse.ArgumentParser(description=" depth images")
    parser.add_argument("--fname", type=str, default=".", help="Path for loading the video")
    parser.add_argument("--fpath", type=str, default=".")
    parser.add_argument("--out", type=str, default="reco")
    parser.add_argument("--pfilter", type=str, default="low")
    parser.add_argument("--method", type=str, default=1)
    parser.add_argument("--freq", type=float, default=0.0)
    args = parser.parse_args()
    main(**vars(args))
