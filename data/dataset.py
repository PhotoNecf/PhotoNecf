#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .data_utils import read_dng


class RawDataset(Dataset):
    def __init__(self, file_csv, file_root='./data/images/', **kwargs):
        self.df = pd.read_csv(file_csv)
        self.file_root = file_root

    def __getitem__(self, idx):
        imgs = []
        path_list = self.df[self.df['sample_id'] == idx]
        for p in path_list['img_path']:
            temp = read_dng(p)
            imgs.append(np.expand_dims(temp, 0))
        device_id = path_list['device_id'].to_list()[0]
        return np.concatenate(imgs, axis=0), idx, device_id

    def __len__(self):
        return len(self.df['sample_id'].unique())

    def load_single(self, p):
        return read_dng(p)