import cv2 as cv
import numpy as np

def mkVisual(single_channel_image, colormap=cv.COLORMAP_TURBO, maxval=None):
    res = single_channel_image
    maxval = res.max() if maxval is None else maxval
    res = cv.convertScaleAbs(res, alpha=255/maxval)
    if colormap is not None:
        res = cv.applyColorMap(res, colormap)
    else:
        res = np.dstack((res,)*3)
    return res

