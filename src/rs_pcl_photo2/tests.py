from cython_module.image_reconstruction import Image_Reconstruction
from numpy import zeros, uint8
image = zeros((10,10), dtype=uint8)
IR = Image_Reconstruction(image)