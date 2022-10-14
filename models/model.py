#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .extract_func import extract_fp_single, extract_fp_multiple
from .ffdnet import FFDNet
from collections import OrderedDict
import torch


class Model(object):
    def __init__(self, model_dir, sigma: float = 5, wdft_sigma: float = 0, device: str = "cuda"):
        self.sigma = sigma
        self.wdft_sigma = wdft_sigma

        state_dict = torch.load(model_dir)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.model', 'model')
            new_state_dict[k] = v

        self.model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()
        self.model.to(device)

    def forward(self, im):
        with torch.no_grad():
            if im.shape[0] == 1:
                im = im.squeeze().numpy()
                return extract_fp_single(im, sigma=self.sigma, wdft_sigma=self.wdft_sigma, model=self.model, device='cuda')
            else:
                im = im.numpy()
                return extract_fp_multiple(im, sigma=self.sigma, model=self.model, device='cuda')

    def __call__(self, im):
        return self.forward(im)