#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
from data import RawDataset
from models import Model
from torch.utils.data import DataLoader
import os
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
from loguru import logger


def extract(input_csv: str, model_file: str, output_dir: str, img_dir: str, concurrency: int = 8):
    dataset = RawDataset(input_csv, file_root=img_dir)
    loader = DataLoader(dataset, batch_size=concurrency)
    logger.info(f"length of dataset is {len(dataset)}")
    model = Model(model_dir=model_file)
    pool = Pool(concurrency)
    for imgs, sample_id, device_id in tqdm(loader):
        res = pool.map(model.forward, imgs)
        sample_id = sample_id.tolist()
        device_id = device_id.tolist()
        for k in range(len(res)):
            p = os.path.join(output_dir, 'cam_{:0>3d}_idx_{:0>6d}.npy'.format(device_id[k], sample_id[k]))
            np.save(p, res[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, default="./data/infos/test_single.csv", help="CSV file containing image infos")
    parser.add_argument("--model-file", type=str, default="./models/ckpts/model.pth", help="Fingerprint extraction network model")
    parser.add_argument("--output-dir", type=str, default="./results/fingerprints/", help="Save fingerprint npy file dir")
    parser.add_argument("--image-dir", type=str, default="./data/images/", help="RAW image file dir")
    parser.add_argument("--concurrency", type=int, default=8, help="multiprocessing workers")

    args = parser.parse_args()
    extract(args.csv_file, args.model_file, args.output_dir, args.image_dir, args.concurrency)

