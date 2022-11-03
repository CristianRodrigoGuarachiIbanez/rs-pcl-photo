import unittest
import cv2 as cv
from rs_pcl_photo2.mk_normals import FFT
from rs_pcl_photo2.lib.depth_image import image_to_array_2d, calc_normals_from_depth
from rs_pcl_photo2.filter import gaussianLP, gaussianHP

class TestFFT(unittest.TestCase):
    def reset(self):
        self._fft = FFT(self._read())

    def _read(self, fname="depth_images/rs-photo-01-depth.png" ):
        return image_to_array_2d(cv.imread(fname, cv.IMREAD_UNCHANGED))

    def reconstruct_image(self):
        new_fft = self._fft.reconstruct_image(file=None, filter=None, mask=None, method=1, freq=0.499211)
        assert new_fft.shape == self._fft.shape
    def reconstruct_with_LPfilter(self):
        filter = self._fft.set_filter()
        mask = gaussianLP(100, filter.shape)
        lowPass = self._fft.reconstruct_image(None, mask, None, 1, 0.0)
        assert lowPass.shape == filter.shape and lowPass.shape == self._fft.getImage().shape


if __name__ == "__main__":
    pass
