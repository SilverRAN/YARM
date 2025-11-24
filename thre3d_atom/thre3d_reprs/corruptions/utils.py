import numpy as np
import torch as ch
from kornia.filters import gaussian as k_gaussian
from kornia.geometry.transform import rotate
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian
import cv2
import torch.nn.functional as F


def half_gaussian(N, stdev):
    assert N % 2 == 1
    rng = ch.arange(N)
    return ch.exp(-ch.pow(rng, 2)/(2.0*stdev**2))

def gaussian_motionfilter2d(kernel_size, stdev, angle):
    if angle.dim() == 0: angle = angle.unsqueeze(dim=0)
    kernel_shape = (angle.shape[0], kernel_size, kernel_size)
    # direction from [-1, 1] to [0, 1] range
    kernel = ch.zeros(kernel_shape, dtype=ch.float)
    kernel[:, kernel_size // 2, :] = half_gaussian(kernel_size, stdev)
    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel[:,None,...], angle, mode='nearest')[:,0]
    return kernel

def torch_to_np(tens):
    return (tens.detach().cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

def np_to_torch(arr):
    return ch.tensor(arr.astype(np.float32).transpose(0, 3, 1, 2) / 255)

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize, wd):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wd

    maparray -= maparray.min()
    return maparray / maparray.max()

# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, c):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

# def rgb2hsv(inp):
#     """Batched version of skimage conversion"""
#     # V channel (B, H, W)
#     out = ch.zeros_like(inp)
#     out_v = inp.max(dim=1).values 

#     # -- S channel
#     delta = inp.max(dim=1).values - inp.min(dim=1).values # (B, H, W)

#     out_s = delta / out_v
#     out_s[delta == 0.] = 0.

#     # -- H channel
#     # red is max
#     idx = (inp[:,0,...] == out_v)
#     out[:,0,...][idx] = (inp[:,1,...][idx] - inp[:,2,...][idx]) / delta[idx]

#     # green is max
#     idx = (inp[:,1,...] == out_v)
#     out[:,0,...][idx] = 2 + (inp[:,2,...][idx] - inp[:,0,...][idx]) / delta[idx]

#     # blue is max
#     idx = (inp[:,2,...] == out_v)
#     out[:,0,...][idx] = 4 + (inp[:,0,...][idx] - inp[:,1,...][idx]) / delta[idx]

#     out_h = (out[:,0,...] / 6.) % 1.
#     out_h[delta == 0.] = 0.

#     # -- output
#     out[:,0,...] = out_h
#     out[:,1,...] = out_s
#     out[:,2,...] = out_v

#     # # remove NaN
#     out[ch.isnan(out)] = 0

#     return out

# def hsv2rgb(inp):
#     hi = ch.floor(inp[:,0,...] * 6)
#     f = inp[:,0,...] * 6 - hi
#     p = inp[:,2,...] * (1 - inp[:,1,...])
#     q = inp[:,2,...] * (1 - f * inp[:,1,...])
#     t = inp[:,2,...] * (1 - (1 - f) * inp[:,1,...])
#     v = inp[:,2,...]

#     hi = ch.stack([hi, hi, hi], axis=1).long() % 6
#     lists = [ch.stack((v, t, p), axis=1),
#              ch.stack((q, v, p), axis=1),
#              ch.stack((p, v, t), axis=1),
#              ch.stack((p, q, v), axis=1),
#              ch.stack((t, p, v), axis=1),
#              ch.stack((v, p, q), axis=1)]
#     out = lists[0]
#     for i in range(1, 6):
#         out = ch.where(hi == i, lists[i], out)
#     return out

def rgb2hsv(rgb: ch.Tensor) -> ch.Tensor:
    eps = 1e-8
    hue = ch.Tensor(rgb.shape[0], rgb.shape[2], rgb.shape[3]).to(rgb.device)
    hue[ rgb[:, 2]==rgb.max(dim=1)[0] ] = 4.0 + ( (rgb[:, 0]-rgb[:, 1]) / (rgb.max(dim=1)[0]-rgb.min(dim=1)[0]+eps) )[ rgb[:, 2]==rgb.max(dim=1)[0] ]
    hue[ rgb[:, 1]==rgb.max(dim=1)[0] ] = 2.0 + ( (rgb[:, 2]-rgb[:, 0]) / (rgb.max(dim=1)[0]-rgb.min(dim=1)[0]+eps) )[ rgb[:, 1]==rgb.max(dim=1)[0] ]
    hue[ rgb[:, 0]==rgb.max(dim=1)[0] ] = (0.0 + ((rgb[:, 1]-rgb[:, 2]) / (rgb.max(dim=1)[0]-rgb.min(dim=1)[0]+eps) )[ rgb[:, 0]==rgb.max(dim=1)[0] ]) % 6.0

    hue[rgb.min(dim=1)[0]==rgb.max(dim=1)[0]] = 0.0
    hue = hue / 6.0

    saturation = (rgb.max(dim=1)[0] - rgb.min(dim=1)[0]) / (rgb.max(dim=1)[0] + eps)
    saturation[rgb.max(dim=1)[0]==0.0] = 0.0

    value = rgb.max(dim=1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = ch.cat([hue, saturation, value], dim=1)
    return hsv

def hsv2rgb(hsv: ch.Tensor) -> ch.Tensor:
    h, s, v = hsv[:, 0, ...], hsv[:, 1, ...], hsv[:, 2, ...]
    h = h % 1.0
    s = ch.clamp(s, 0.0, 1.0)
    v = ch.clamp(v, 0.0, 1.0)

    r = ch.zeros_like(h)
    g = ch.zeros_like(h)
    b = ch.zeros_like(h)

    hi = ch.floor(h * 6.0)
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0], g[hi0], b[hi0] = v[hi0], t[hi0], p[hi0]
    r[hi1], g[hi1], b[hi1] = q[hi1], v[hi1], p[hi1]
    r[hi2], g[hi2], b[hi2] = p[hi2], v[hi2], t[hi2]
    r[hi3], g[hi3], b[hi3] = p[hi3], q[hi3], v[hi3]
    r[hi4], g[hi4], b[hi4] = t[hi4], p[hi4], v[hi4]
    r[hi5], g[hi5], b[hi5] = v[hi5], p[hi5], q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = ch.cat([r, g, b], dim=1)
    return rgb