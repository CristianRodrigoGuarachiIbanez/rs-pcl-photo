
from rs_pcl_photo2.lib.depth_image import image_to_array_2d, calc_normals_from_depth
from rs_pcl_photo2.td_fft.load_img import *
from rs_pcl_photo2.td_fft.fourier_mask import *
from rs_pcl_photo2.td_fft.image_reconstruction import *
from rs_pcl_photo2.cython_module.image_reconstruction import Image_Reconstruction
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pytest


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


@pytest.mark.parametrize("fname", [("depth_images/rs-photo-00-depth.png")])
def test_rs_pcl_photo(fname):

    dimg = image_to_array_2d(cv.imread(fname, cv.IMREAD_UNCHANGED))
    array_size = min(dimg.shape) - 1 + min(dimg.shape) % 2

    # Crop image so it's a square image
    centre = int((array_size - 1) / 2)

    # Array dimensions (array is square) and centre pixel
    coords_left_half = square_image(dimg, array_size, centre)
    ft = calculate_2dft(dimg)
    # Reconstruct image

    IR = Image_Reconstruction(dimg, np.asarray(coords_left_half, dtype='int32'), ft, centre)
    com = IR.individualGrating()
    print(com.shape)
    plt.savefig("./output")


if __name__ == "__main__":
    from rs_pcl_photo2.lib.util import mkVisual
    import sys
    from functools import partial

    mkVisual = partial(mkVisual, colormap=cv.COLORMAP_CIVIDIS)
    fname = sys.argv[1] if len(sys.argv) > 1 else "depth_images/rs-photo-00-depth.png"
    dimg = image_to_array_2d(cv.imread(fname, cv.IMREAD_UNCHANGED))

    # Array dimensions (array is square) and centre pixel
    # Use smallest of the dimensions and ensure it's odd
    array_size = min(dimg.shape) - 1 + min(dimg.shape) % 2

    # Crop image so it's a square image
    centre = int((array_size - 1) / 2)

    # Array dimensions (array is square) and centre pixel
    coords_left_half = square_image(dimg, array_size, centre)
    ft = calculate_2dft(dimg)
    # Reconstruct image

    IR = Image_Reconstruction(dimg, np.asarray(coords_left_half, dtype='int32'), ft, centre)
    com = IR.individualGrating()
    print(com.shape)
    plt.savefig("./output")
    """fig = plt.figure()

    # Step 1
    # Set up empty arrays for final image and individual gratings
    rec_image = np.zeros(dimg.shape)
    individual_grating = np.zeros(dimg.shape, dtype="complex")

    idx = 0
    # Step 2
    for coords in coords_left_half:
        # Central column: only include if points in top half of the central column
        if not (coords[1] == centre and coords[0] > centre):
            idx += 1
            symm_coords = find_symmetric_coordinates(coords, centre)

            # Step 3
            # Copy values from Fourier transform into individual_grating for the pair of points in
            # current iteration
            individual_grating[coords] = ft[coords]
            individual_grating[symm_coords] = ft[symm_coords]
            # Step 4
            # Calculate inverse Fourier transform to give the reconstructed grating. Add this reconstructed
            # grating to the reconstructed image
            rec_grating = calculate_2dift(individual_grating)
            print("symm -> ", type(rec_grating), rec_grating.dtype, rec_grating.shape)
            exit()
            rec_image += rec_grating
            # Clear individual_grating array, ready for next iteration
            individual_grating[coords] = 0
            individual_grating[symm_coords] = 0
            print(" ind -> ",individual_grating[coords], "index -> ", idx)
            # display_plots(rec_grating, rec_image, idx)
    # fig.savefig("./figure.png")
    # plt.show()"""