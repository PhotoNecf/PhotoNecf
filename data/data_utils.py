#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rawpy
import numpy as np
from typing import Tuple

def cut_center(img, sizes: Tuple) -> np.ndarray:
    array = np.copy(img)
    if array.ndim == 3 and len(sizes) == 2:
        sizes = (*sizes, array.shape[2])
    elif (array.ndim != len(sizes)):
        raise ValueError(f"array.ndim {array.ndim} are not equal to sizes {len(sizes)}")
    for axis in range(array.ndim):
        axis_target_size = sizes[axis]
        axis_original_size = array.shape[axis]
        if axis_target_size > axis_original_size:
            raise ValueError(
                'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                                                                      axis_original_size))
        else:
            axis_start_idx = (axis_original_size - axis_target_size) // 2
            axis_end_idx = axis_start_idx + axis_target_size
            array = np.take(array, np.arange(axis_start_idx, axis_end_idx), axis)
    return array


def read_dng(file_path: str, crop_length: int = 2048, bit_num: int = 12) -> np.ndarray:
    with rawpy.imread(file_path) as raw:
        im_raw = raw.raw_image.copy()
    if im_raw.shape[0] > im_raw.shape[1]:
        im_raw = np.rot90(im_raw)
    img = (im_raw / 2 ** (bit_num - 8)).astype(np.uint8)
    return cut_center(img, (crop_length, crop_length))