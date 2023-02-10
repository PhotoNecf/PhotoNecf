import numpy as np
import torch

from multiprocessing import Pool
from typing import List, Tuple, Any, Dict, Union
from functools import partial
from scipy.ndimage import filters
from numpy.fft import fft2, ifft2
import pywt

def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)


def zero_mean(im: np.ndarray) -> np.ndarray:
    """
    zero-mean to normalize pixel value distribution
    :param im: input image
    :return: zero-mean of result
    """
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    im_zm = im - ch_mean

    # Compute the 1D mean on both x and y axis, then subtract
    row_mean = im_zm.mean(axis=1)
    col_mean = im_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    im_zm_r = im_zm - row_mean
    im_zm_rc = im_zm_r - col_mean

    if im.shape[2] == 1:
        im_zm_rc.shape = im.shape[:2]

    return im_zm_rc


def zero_mean_bayer(im: np.ndarray) -> np.ndarray:
    """
    zero-mean process on RGGB pixels
    :param im:
    :return:
    """
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im

def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    noise variance threshold
    :param wlet_coeff_energy_avg: input energy
    :param noise_var: thresholding noise
    :return: energy residual component
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    Wiener adaptive filter to obtain the noise component
    :param x: input image array
    :param noise_var: Power spectral density of the extracted noise
    :param kwargs: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy, window_size, mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    # Wiener filter of noise component
    return x * noise_var / (coef_var_min + noise_var)


def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    final step in PRNU by transform ZM into Fourier domain, filter it using Wiener filter,
    and only keep the noise component. illustated in Section III.B
    :param im: ZM output image
    :param sigma: estimated noise power
    :return: filter result
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_fft = fft2(im)
    im_fft_mag = np.abs(im_fft / (h * w) ** 0.5)

    im_fft_mag_noise = wiener_adaptive(im_fft_mag, noise_var)

    zeros_y, zeors_x = np.nonzero(im_fft_mag == 0)

    im_fft_mag[zeros_y, zeors_x] = 1
    im_fft_mag_noise[zeros_y, zeors_x] = 0

    im_fft_filt = im_fft * im_fft_mag_noise / im_fft_mag
    im_filt = np.real(ifft2(im_fft_filt))

    return im_filt.astype(np.float32)


def wiener_post(noise: np.ndarray,
                wdft_sigma: float = 0) -> np.ndarray:
    W = zero_mean_bayer(noise)
    W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
    W = wiener_dft(W, W_std).astype(np.float32)

    return W


def extract_prnu_single(noise_init: np.ndarray,
                        concurrency: int=16) -> np.ndarray:
    """
    extract prnu from a single image
    :param im: grayscale or color image, np.uint8 (jpg) or np.uint16 (raw)
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: PRNU
    """
    pool = Pool(concurrency)
    fp_list = pool.map(wiener_post, noise_init)
    return np.array(fp_list)

