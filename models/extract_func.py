#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
from typing import List, Tuple, Any, Dict, Union
from scipy.ndimage import filters
from numpy.fft import fft2, ifft2


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


def noise_extract(im: np.ndarray, sigma: float = 5, model = None, device = "cpu") -> np.ndarray:
    """
    noise extraction as Section.3 of our paper
    :param im: grayscale or color image, np.uint8 or np.uint16
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual, i.e., original_image - denoised image or high frequency component of original image
    """
    if im.dtype != np.uint8 and im.dtype != np.uint16:
        raise ValueError(f"dtype of image input in noise extract module is neither uint8 nor uint16.")

    if im.ndim not in [2, 3]:
        raise ValueError(f"Unavailable image input dimension = {im.ndim} in noise extract module. should be 2 or 3.")

    if im.dtype == np.uint8:
        im = im.astype(np.float32)
    else:
        im = (im / 256).astype(np.float32)

    noisy_img = im / 255.
    if device == "cpu":
        input_img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(noisy_img), 0), 0).float()
        input_sigma = torch.full((1, 1, 1, 1), sigma / 255.).type_as(input_img)
        with torch.no_grad():
            res_noise = model(input_img, input_sigma)
    else:
        input_img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(noisy_img), 0), 0).float()
        input_sigma = torch.full((1, 1, 1, 1), sigma / 255.).type_as(input_img)
        with torch.no_grad():
            res_noise = model(input_img.to(device), input_sigma.to(device))
            res_noise = res_noise.cpu()
    W = res_noise.squeeze().numpy() * 255

    return W


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


def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    final step by transform ZM into Fourier domain, filter it using Wiener filter,
    and only keep the noise component.
    :param im: ZM output image
    :param sigma: estimated noise power
    :return: filter result
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_fft = fft2(im)
    # div sqrt dim to keep spatial and spectrum energy the same
    im_fft_mag = np.abs(im_fft / (h * w) ** 0.5)

    im_fft_mag_noise = wiener_adaptive(im_fft_mag, noise_var)

    # get the positions of zero elements in im_fft_mag
    zeros_y, zeors_x = np.nonzero(im_fft_mag == 0)

    im_fft_mag[zeros_y, zeors_x] = 1
    im_fft_mag_noise[zeros_y, zeors_x] = 0

    im_fft_filt = im_fft * im_fft_mag_noise / im_fft_mag
    im_filt = np.real(ifft2(im_fft_filt))

    return im_filt.astype(np.float32)

def inten_scale(im: np.ndarray, threshold: int = 252, v: int = 6) -> np.ndarray:
    """
    translate int pixel value to float, filter the saturated pixel value
    :param im: input uint8 or uint16 image
    :param threshold: saturated threshold value
    :param v: adjust value over threshold
    :return: intensity adjusted version of input x
    """

    assert (im.dtype == np.uint8 or im.dtype == np.uint16)

    if im.dtype != np.uint8:
        im = (im / 256).astype(np.uint8)

    out = np.exp(-1 * (im - threshold) ** 2 / v)
    out[im < threshold] = im[im < threshold] / threshold

    return out


def saturation(im: np.ndarray, threshold: int = 250) -> np.ndarray:
    """
    obtain overexposure areas as False and regular areas as True
    :param im: input uint8 or uint16 image
    :param threshold: saturated threshold value
    :return: saturation map from input im
    """
    assert (im.dtype == np.uint8 or im.dtype == np.uint16)

    if im.dtype != np.uint8:
        im = (im / 256).astype(np.uint8)

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    if im.max() < threshold:
        return np.squeeze(np.ones((h, w, ch)))

    im_h = im - np.roll(im, (0, 1), (0, 1))
    im_v = im - np.roll(im, (1, 0), (0, 1))
    satur_map = \
        np.bitwise_not(
            np.bitwise_and(
                np.bitwise_and(
                    np.bitwise_and(
                        im_h != 0, im_v != 0
                    ), np.roll(im_h, (0, -1), (0, 1)) != 0
                ), np.roll(im_v, (-1, 0), (0, 1)) != 0
            )
        )

    max_ch = im.max(axis=0).max(axis=0)

    for ch_idx, max_c in enumerate(max_ch):
        if max_c > threshold:
            satur_map[:, :, ch_idx] = \
                np.bitwise_not(
                    np.bitwise_and(
                        im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                    )
                )

    return np.squeeze(satur_map)

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

def extract_fp_single(im: np.ndarray,
                        sigma: float = 5,
                        wdft_sigma: float = 0,
                        model = None,
                        device = "cpu") -> np.ndarray:
    """
    extract fingerprint from a single image
    :param im: grayscale or color image, np.uint8 (jpg) or np.uint16 (raw)
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: fingerprint or residual noise
    """
    W = noise_extract(im, sigma, model=model, device=device)
    W = rgb2gray(W)
    W = zero_mean_bayer(W)
    W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
    W = wiener_dft(W, W_std).astype(np.float32)
    # W = W.clip(-1,1)
    return W

def extract_fp_multiple(imgs: Union[List[np.ndarray], np.ndarray],
                          sigma: float = 5,
                          model = None,
                          device = "cpu") -> np.ndarray:
    """
    extract fingerprint from multiple images
    :param imgs: list of images of size (H,W,Ch)
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: fingerprint or residual noise
    """
    if not isinstance(imgs[0], np.ndarray):
        raise ValueError(f"dtype of image input List should be numpy arrays.")

    if imgs.ndim not in [3, 4]:
        raise ValueError(f"Unavailable image input dimension = {imgs.ndim} in noise extract module. should be 3 or 4.")

    if imgs.ndim == 4:
        h, w, ch = imgs[0].shape
        RPsum = np.zeros((h, w, ch), np.float32)
        NN = np.zeros((h, w, ch), np.float32)
    else:
        h, w = imgs[0].shape
        RPsum = np.zeros((h, w), np.float32)
        NN = np.zeros((h, w), np.float32)

    for im in imgs:
        RPsum += noise_extract(im, sigma, model=model, device=device)
        NN += (inten_scale(im) * saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_bayer(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K



