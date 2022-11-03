#!/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2 as cv

from scipy.signal import convolve2d
from lib import gltfgen
from lib.depth_image import (image_to_array_2d, array_2d_to_image, calc_normals_from_depth)
from lib.util import mkVisual


FPS = 30
WIDTH = 1280
HEIGHT = 720
MIN_DEPTH = 10
N_MEASUREMENT_FOR_FRAME = 4

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)


pl = rs.pipeline()
prof = pl.start(cfg)

d_sensor = prof.get_device().first_depth_sensor()
d_scale = d_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

hf_flt = rs.hole_filling_filter(1)
dec_flt = rs.decimation_filter(2.0)

# _array_info = lambda a: f"{a.shape}, {a.dtype}, min: {a.min()}, max: {a.max()}"


def _array_info(a):
    return f"{a.shape}, {a.dtype}, min: {a.min()}, max: {a.max()}"


def record_normals(name, dimg):
    ns = calc_normals_from_depth(dimg)
    cv.imwrite(f"{name}-normals.png", (ns*255).astype("uint8"))


def record_pcl(name, dimg, cimg, intrinsics=None):
    x, y, z = (a.flatten() for a in dimg_to_pcl(dimg, intrinsics))
    b, g, r = (a.flatten() for a in np.dsplit(cimg, 3))
    a = np.full(b.shape, 255, 'uint8')

    points = np.stack((x, y, z)).T
    colors = np.stack((r, g, b, a)).T

    if intrinsics is None:
        cam = gltfgen.mkOrthographicCamera(zfar=100)
    else:
        aspectRatio = intrinsics.width / intrinsics.height
        yfov = 2 * np.arctan2(intrinsics.height, intrinsics.fy)
        cam = gltfgen.mkPerspectiveCamera(aspectRatio=aspectRatio, yfov=yfov, znear=0.01, zfar=100)

    gltfgen.create_gltf_pcl(points, colors, name)

    print(f'recorded {name}')


def record_photos(name, dimg, cimg, dintr):
    print(f"cimg: " + _array_info(cimg))
    print(f"dimg: " + _array_info(dimg))
    cv.imwrite(f"{name}-color.png", cimg)
    cv.imwrite(f"{name}-depth.png", _img := array_2d_to_image(dimg))

    ## for debugging: can the depth array be recontructed successfully ->
    ## *-depth-orig.png == *-depth-reloaded.png (visually/rounded)
    print(f"_img: " + _array_info(_img))
    cv.imwrite(f"{name}-depth-orig.png", mkVisual((dimg)))
    cv.imwrite(f"{name}-depth-reloaded.png", mkVisual(image_to_array_2d(_img)))


def dimg_to_pcl(dimg, intrinsics=None):
    h, w = dimg.shape
    nx, ny = (np.linspace(0, d-1, d) for d in (w,h))
    u, v = np.meshgrid(nx, ny)

    if intrinsics is not None:
        xs = (u - intrinsics.ppx) / intrinsics.fx
        ys = (v - intrinsics.ppy) / intrinsics.fy
    else:
        xs, ys = u, v

    if intrinsics is not None:
        zs = dimg / 1000 # to meters
        xs, ys = ( np.multiply(ds, zs) for ds in (xs, ys))
    else:
        zs = dimg
    return xs, ys, zs


def average_dimg(dimgs):
    res = np.dstack(dimgs)
    valid_count = np.sum((res > 0) * 1.0, axis=2)
    valid_count = np.asarray(valid_count, dtype='float32')
    res = res.sum(axis=2)
    res = np.divide(res, valid_count, where=valid_count > 0)
    np.putmask(res, valid_count < 1, 0)
    return res, valid_count


def convolve2d_all_zeros(img, kernel, lim=float('inf')):
    res = img
    i = 0
    while np.count_nonzero(res) < res.size and i < lim:
        res = convolve2d(res, np.rot90(kernel), fillvalue=1, mode='same')
        i += 1
        print(i)
    return res


def remove_black(dimg):
    kernel = np.array([[.3, .3, .3],
                       [ 0,  0,  0],
                       [ 0,  0,  0]])

    res = []
    for _ in range(4):
        kernel = np.rot90(kernel)
        res.append(convolve2d_all_zeros(dimg, kernel, 20))

    return np.max(res, axis=0)


def run():
    rec_cnt = 0
    cv.namedWindow('Panda', cv.WINDOW_GUI_EXPANDED | cv.WINDOW_KEEPRATIO )
    while True:

        dimgs = []
        while len(dimgs) < N_MEASUREMENT_FOR_FRAME:
            frms = pl.wait_for_frames()
            frms = align.process(frms)
            dframe = frms.get_depth_frame()
            cframe = frms.get_color_frame()
            if dframe is None or cframe is None:
                print(f"{cframe=} {dframe=}")
                continue

            dframe = hf_flt.process(dframe)

            dimg = np.asanyarray(dframe.get_data())
            dimg = np.asarray(dimg, dtype='float32')
            dimgs.append(dimg)

        dimg, valid_count = average_dimg(dimgs)

        # print(dimg.min(), dimg.max(), dimg.size - np.count_nonzero(dimg), "||",
                # valid_count.min(), valid_count.max(),
                # valid_count.size - np.count_nonzero(valid_count))

        # dimg = remove_black(dimg)

        cimg = np.asanyarray(cframe.get_data())

        # cv.imshow('valid_count', mkVisual(valid_count))
        cv.imshow('Panda', np.vstack((mkVisual(dimg, maxval=4_500),
                                      cimg,
                                      mkVisual(valid_count, colormap=None))))

        key = cv.waitKey(1) & 0xff
        if key in (27, ord('q')):
            break

        if key in (ord(c) for c in 'Rr'):
            name = f'rs-photo-{rec_cnt:02d}'
            record_pcl(name + '-perspective', dimg, cimg, dintr)
            record_pcl(name + '-orthographic', dimg, cimg)
            record_photos(name, dimg, cimg, dintr)
            record_normals(name, dimg)
            rec_cnt += 1


    cv.destroyAllWindows()


_get_intrinsics = lambda frame: \
    frame.get_profile().as_video_stream_profile().get_intrinsics()


def get_intrinsics():
    frms = pl.wait_for_frames()
    return [_get_intrinsics(f)\
            for f in (frms.get_depth_frame(), frms.get_color_frame())]


def reset():
    global prof
    pl.stop()
    prof.get_device().hardware_reset()
    prof = pl.start(cfg)


if __name__ == "__main__":
    # reset()
    while True:
        try:
            dintr, cintr = get_intrinsics()
            break
        except RuntimeError as e:
            print(e)
            reset()

    try:
        run()
    finally:
        pl.stop()
