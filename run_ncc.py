#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import os
from tqdm import tqdm
import torch
from utils import aligned_cc_torch, gt, stats
import argparse
from loguru import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint-dir", type=str, default="./results/fingerprints",
                        help="Registered fingerprint directory")
    parser.add_argument("--imagenoise-dir", type=str, default="./results/fingerprints",
                        help="Image residual noise directory")
    parser.add_argument("--batch", type=int, default=2048, help="batch load of image noise")
    parser.add_argument("--output-dir", type=str, default="./results/measures/",
                        help="Save fingerprint npy file dir")

    args = parser.parse_args()

    path_k = args.fingerprint_dir
    path_w = args.imagenoise_dir

    name_k = sorted(os.listdir(path_k))
    name_w = sorted(os.listdir(path_w))

    id_k = [n.split('_')[1] for n in name_k]
    id_w = [n.split('_')[1] for n in name_w]

    list_k = [os.path.join(path_k, i) for i in tqdm(name_k)]
    list_w = [os.path.join(path_w, i) for i in tqdm(name_w)]

    logger.info(f"Registered device num is {len(list_k)}, compared image noise residual num is {len(list_w)}")

    data_k = [torch.from_numpy(np.load(p, mmap_mode='r')) for p in (list_k)]
    data_k_stack = torch.stack(data_k, 0)
    ncc_list = []
    for k in (range(0, len(list_w), args.batch)):
        r = k + args.batch
        r = min(r, len(list_w))
        data_w_temp = [torch.from_numpy(np.load(p, mmap_mode='r')) for p in list_w[k:r]]
        data_w_stack_temp = torch.stack(data_w_temp, 0)
        ncc = aligned_cc_torch(data_k_stack, data_w_stack_temp)['ncc']
        ncc_list.append(ncc)

    logger.info(f"finish loading fingerprints and image noises")

    cc_aligned_rot = torch.cat(ncc_list, 1).numpy()
    gt_array = gt(id_k, id_w)
    stats_result = stats(cc_aligned_rot, gt_array)

    save_dict = {
        'path_k':path_k,
        'path_w':path_w,
        'id_k':id_k,
        'id_w':id_w,
        'name_k':name_k,
        'name_w':name_w,
        'cc_aligned_rot':cc_aligned_rot,
        'gt_array':gt_array,
        'stats_result':stats_result
    }
    path_save = os.path.join(args.output_dir, path_k.split('/')[-1] + '_vs_' + path_w.split('/')[-1] + '.npy')
    np.save(path_save, save_dict)

    logger.info(f"Results locates at：{os.path.abspath(path_save)}, "
                f"AUC: {stats_result['auc']*100}"
                f"EER: {stats_result['eer']*100}")

