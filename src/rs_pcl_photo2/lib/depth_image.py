import numpy as np


def image_rgb_to_array_2d(xs):
    def from_bytes(xs):
        for ij in np.ndindex(xs.shape[0:2]):
            bs = xs[ij].astype('uint8').tobytes()
            yield int.from_bytes(bs, 'big')
    return np.fromiter(from_bytes(xs), dtype=float).reshape(xs.shape[0:2])

def array_2d_to_image_rgb(xs):
    def to_bytes(xs):
        for x in np.nditer(xs):
            for c in int(x).to_bytes(3, 'big'):
                yield c
    return np.fromiter(to_bytes(xs), dtype="uint8").reshape(xs.shape + (3,))


SCALE = 10.0

def array_2d_to_image_gray(xs):
    return \
    (xs * SCALE).astype('uint16')

def image_gray_to_array_2d(xs):
    return \
    xs.astype('float') / SCALE




image_to_array_2d = image_gray_to_array_2d
array_2d_to_image = array_2d_to_image_gray

def calc_normals_from_depth(dimg):
    dimg_dx, dimg_dy = np.gradient(dimg)
    vdx = np.dstack((np.ones_like(dimg_dx), np.zeros_like(dimg_dx), dimg_dx))
    vdy = np.dstack((np.zeros_like(dimg_dy), np.ones_like(dimg_dy), dimg_dy))

    cs = np.cross(vdx, vdy)
    norms = np.linalg.norm(cs, axis=2)
    return cs / np.dstack((norms,)*3)
