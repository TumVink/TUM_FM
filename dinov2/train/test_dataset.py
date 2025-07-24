# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# this file was changed

import sys
import os

# Add the root directory of the project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import argparse
import logging
import math
import os
from functools import partial
import wandb
#os.environ["WANDB_MODE"]="offline" #use this so set wandb to offline mode

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
import torch.distributed as dist
from dinov2.data.datasets.CustomImageDataset import save_all

from dinov2.train.ssl_meta_arch import SSLMetaArch
#os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "ibp170s0f0"


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="/home/ge24juj/dino-tum/dinov2/configs/ssl_default_config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser



def viz_data(cfg): # change resume to true?

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )
    inputs_dtype = torch.half
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=0,  # TODO: Fix this -- cfg.train.seed
        drop_last=True,
        collate_fn=collate_fn,
    )

    #iterate over the torch dataloader and save the first 10 images
    global_num = 2
    local_num = 2
    for batch_idx, data in enumerate(data_loader):
        pass
        # print(f'Batch {batch_idx + 1}')
        # img_pil = {}
        # img_pil['global_crops'] = data['collated_global_crops']
        # print(data['collated_global_crops'].shape)
        # img_pil['local_crops'] = data['collated_local_crops']
        # save_all(img_pil,f'Batch{batch_idx + 1}',global_num,local_num)
        # break
        # print('Data:', data)
        # print('Labels:', labels)





def main(args):
    cfg = setup(args)

    viz_data(cfg)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
