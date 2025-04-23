#From: UNDERSTANDING DIMENSIONAL COLLAPSE IN CONTRASTIVE SELF-SUPERVISED LEARNING
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import random
import shutil
import sys
import time
import warnings
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Visualize spectrum')
parser.add_argument('--data', metavar='DIR', default="/datasets01/imagenet_full_size/061417",
                    help='path to dataset')
parser.add_argument('--rep', action="store_true")
parser.add_argument('--projector', action="store_true")
parser.add_argument('--checkpoint', type=str)


def main(feats_dir_ls):
    emd_ls = []
    for i in range(len(feats_dir_ls)):
        feats_dir = feats_dir_ls[i]
        embedding_spectrum = singular(feats_dir)
        emd_ls.append(np.log(embedding_spectrum))
        print(np.log(embedding_spectrum))

    plt_embedding_spectrum(emd_ls,feats_dir_ls)

def singular(feats_dir):
    latents = torch.load(feats_dir)
    z = torch.nn.functional.normalize(latents, dim=1)

    # calculate covariance
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)

    return d


def exclude_bias_and_norm(p):
    return p.ndim == 1

def plt_embedding_spectrum(emd_ls,feats_dir_ls):
    assert len(emd_ls) == 8
    #plot and save the embedding spectrum
    #set up figure size

    fig, ax = plt.subplots(2,4)
    fig.set_size_inches(40, 20)
    for i in range(2):
        for j in range(4):
            ax[i,j].plot(emd_ls[i*4+j])
            ax[i,j].set_ylim([-25, 0])
            ax[i,j].set_xlabel('Singular Value Rank Index')
            ax[i,j].set_ylabel('Log of singular values')
            ax[i,j].set_title(feats_dir_ls[i*4+j].split('/')[-1].split('.')[0].replace('Dino_manual_74340','TUM'))

    plt.savefig('embedding_spectrum.png')



    #plt.plot(np.log(embedding_spectrum))
    #plt.xlabel('Singular Value Rank Index')
    #plt.ylabel('Log of singular values')

    #plt.savefig('embedding_spectrum.png')

if __name__ == '__main__':
    feats_dir_ls = ["/home/ge54xof/dino-tum/dinov2/downstream/feats/CRC_norm_UNI_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/CRC_UNI_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/MHIST_UNI_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/Patch_cam_UNI_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/CRC_norm_Dino_manual_74340_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/CRC_Dino_manual_74340_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/MHIST_Dino_manual_74340_test.pth",
                    "/home/ge54xof/dino-tum/dinov2/downstream/feats/Patch_cam_Dino_manual_74340_test.pth"]




    main(feats_dir_ls)