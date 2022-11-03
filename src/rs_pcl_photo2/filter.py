import numpy as np


def distance(point1, point2):
    return np.sqrt((point1[0] - point1[1])**2 + (point2[0] - point2[1])**2)

def gaussianLP(DO, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y,x), center)**2)/(2*(DO**2))))
    return base
def gaussianHP(DO, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - np.exp(((-distance((y, x), center) ** 2) / (2 * (DO ** 2))))