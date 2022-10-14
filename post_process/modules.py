#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import torch


def get_weight_mask(im, patch_size=128, top_num=64, upper=255, weighted=False):
    im_patch = im.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

    im_mean = im_patch.float().mean([2, 3])
    im_mean[im_mean >= upper] *= -1

    val, _ = im_mean.flatten().topk(top_num)
    th = val[-1]
    peak = val[0]

    im_mean[im_mean < th] = 0
    if weighted:
        im_mean[im_mean >= th] /= peak
    else:
        im_mean[im_mean >= th] = 1

    mask = im_mean.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return mask

def binarize(im):
    a = torch.ones_like(im)
    b = -torch.ones_like(im)
    output = torch.where(im >= 0, a, b)
    return output