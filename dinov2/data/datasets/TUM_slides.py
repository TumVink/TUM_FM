# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset
import pandas as pd


logger = logging.getLogger("dinov2")
_Target = int

def sample_patches_from_csv(total_batches = 5):
    '''
        Sample_patches id from the first 5 patches
        num_batches: int
        return: df of patches for the first total_batches
    '''
    # read the csv file in dir /home/ge54xof/Foundation-Model-for-Pathology/data/, with name local_slides_df_bt_+'num_batches'+.csv

    local_slides_dfs = [os.path.join('/home/ge54xof/Foundation-Model-for-Pathology/data/local_slides_df_bt_'+str(nr_batch)+'.csv') for nr_batch in range(total_batches)]
    if len(local_slides_dfs) == 0:
        local_slides_df = None
    else:
        local_slides_df = pd.concat([pd.read_csv(df) for df in local_slides_dfs])
        # exclude the items where the 'status' is 'corrupted'
        local_slides_df = local_slides_df[local_slides_df['status'] != 'corrupted']
        local_slides_paths_ls = local_slides_df['slide_id'].tolist()
        local_num_patches_ls = local_slides_df['num_patches'].tolist()

        local_slides_df = {
                    'slide_id': local_slides_paths_ls,
                    'status': ['tb trained'*len(local_num_patches_ls)],  # 'deleted', 'training', 'tb trained'
                    'num_patches':local_num_patches_ls

                }


    return local_slides_df

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 5000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class TUM_slides(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        target_patch_size=None, patches_per_slide=500,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        super().__init__(root, transforms, transform, target_transform)
        self.target_patch_size = target_patch_size
        self.patches_per_slide = patches_per_slide
        self.transform = transform
        self.slide_ext = '.h5'
        # self.data_h5_path = "/mnt/data/"

        self.local_slides_df = sample_patches_from_csv()

        self.num_ls = self.local_slides_df.num_patches.to_numpy()
        self.length = self.num_ls.sum()
        self.accumulate_ls = np.add.accumulate(self.num_ls)
        self.slide_id_ls = self.local_slides_df.slide_id.tolist()

        #self.slides_status.to_csv('slides_status.csv', index=False)
        #np.random.seed(0)

    @property
    def split(self) -> "ImageNet.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)


    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]

        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]


    def __len__(self):
        return self.length # todo

    def __getitem__(self, idx):

        # select the patch
        bag_candidate_idx = np.argwhere(self.accumulate_ls > idx).min()    #]
        if bag_candidate_idx>0:
            patch_local_idx = idx - self.accumulate_ls[bag_candidate_idx - 1]
        else:
            patch_local_idx = idx

        slide_id = self.slide_id_ls[bag_candidate_idx]
        #print('Fking some error happens here with .loc()')
        slide_dir = slide_id

        with h5py.File(slide_dir, "r") as f:
            #try:
            img = Image.fromarray(f['imgs'][patch_local_idx],mode="RGB")#.convert("RGB") #

        return img



