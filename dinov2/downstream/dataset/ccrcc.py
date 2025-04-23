import os
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
# import pathlib
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import _decompress, download_file_from_google_drive, verify_str_arg

from PIL import Image
import pandas as pd
import glob
from tqdm import tqdm
from collections import OrderedDict


class DatasetMHIST(VisionDataset):
    def __init__(
            self,
            root: str = '/mnt/nfs03-R6/mhist/',
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,):

        """
        MHIST dataset class wrapper (train with augmentation)
        """

        #self.image_size = image_size

        # Resize images
        if split == 'val':
            split = 'valid'
        self._split = verify_str_arg(split, "split", ("train", "test", "valid"))
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(root,'mhist_'+self._split+'.h5')
        self.cache_img = OrderedDict()
        self.cache_tgt = OrderedDict()
        self.max_cache_length = 4

        # Data augmentations
        # self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        # self.transform5 = Compose(
        #     [Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
        #      Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

        # GT annotation
        #GT = pd.read_csv(annot_path, header=None)

        # self.datalist = []
        # self.num_pos = 0
        # self.num_neg = 0
        # img_paths = glob.glob('{}/*.png'.format(dataset_path))
        # with tqdm(enumerate(sorted(img_paths)), disable=True) as t:
        #     for wj, img_path in t:
        #         head, tail = os.path.split(img_path)
        #         img_id = tail  # Get image_id
        #
        #         # check if it belongs to train/val set
        #         set = GT.loc[GT[0] == img_id][3]
        #         label = GT.loc[GT[0] == img_id][1]
        #
        #         # Add only train/test to the corresponding set
        #         if set.iloc[0] == 'train':
        #             if label.iloc[0] == 'HP':
        #                 cls_id = 0
        #                 self.num_neg +=1
        #             else:
        #                 cls_id = 1  # SSA
        #                 self.num_pos +=1
        #             self.datalist.append((img_path, cls_id))
        #         else:
        #             continue


    def __len__(self):
        #images_file = self._FILES[self._split]["images"][0]
        with h5py.File(self.img_dir) as images_data:
            return images_data["images"].shape[0]

    def __getitem__(self, idx):

        #images_file = self._FILES[self._split]["images"][0]
        if self.img_dir in self.cache_img:
            image = Image.fromarray(self.cache_img[self.img_dir][idx]).convert("RGB")
            #target = int(targets_data["y"][idx, 0, 0, 0])
        else:
            if len(self.cache_img) > self.max_cache_length:
                # dictionay remove the first added item
                self.cache.popitem(last=False)
            with h5py.File(self.img_dir) as images_data:
                self.cache_img[self.img_dir] = images_data["images"][:]
                image = Image.fromarray(images_data["images"][idx]).convert("RGB")


        if self.img_dir in self.cache_tgt:
            target = int(self.cache_tgt[self.img_dir][idx])
        else:
            if len(self.cache_tgt) > self.max_cache_length:  # shape is [num_images, 1, 1, 1]
                # dictionay remove the first added item
                self.cache.popitem(last=False)
            with h5py.File(self.img_dir) as targets_data:
                self.cache_tgt[self.img_dir] = targets_data["labels"][:]
                target = int(targets_data["labels"][idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

class DatasetMHIST_train():
    def __init__(self, dataset_path = '/mnt/mhist/images/', annot_path='/mnt/mhist/annotations.csv', image_size=224,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):

        """
        MHIST dataset class wrapper (train with augmentation)
        """

        #self.image_size = image_size

        # Resize images
        #self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])
        self.transform = transform
        self.target_transform = target_transform

        # Data augmentations
        # self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        # self.transform5 = Compose(
        #     [Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
        #      Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

        # GT annotation
        GT = pd.read_csv(annot_path, header=None)

        self.datalist = []
        self.num_pos = 0
        self.num_neg = 0
        img_paths = glob.glob('{}/*.png'.format(dataset_path))
        with tqdm(enumerate(sorted(img_paths)), disable=True) as t:
            for wj, img_path in t:
                head, tail = os.path.split(img_path)
                img_id = tail  # Get image_id

                # check if it belongs to train/val set
                set = GT.loc[GT[0] == img_id][3]
                label = GT.loc[GT[0] == img_id][1]

                # Add only train/test to the corresponding set
                if set.iloc[0] == 'train':
                    if label.iloc[0] == 'HP':
                        cls_id = 0
                        self.num_neg +=1
                    else:
                        cls_id = 1  # SSA
                        self.num_pos +=1
                    self.datalist.append((img_path, cls_id))
                else:
                    continue


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        image = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        target = int(self.datalist[index][1])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def get_stats(self):
        return self.num_pos,self.num_neg





class DatasetMHIST_test():
    def __init__(self, dataset_path='/mnt/mhist/images/', annot_path='/mnt/mhist/annotations.csv',
                 image_size=224,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):

        """
        MHIST dataset class wrapper (train with augmentation)
        """

        #self.image_size = image_size

        # Resize images
        # self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])
        self.transform = transform
        self.target_transform = target_transform
        self.num_pos = 0
        self.num_neg = 0

        # Data augmentations
        # self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        # self.transform5 = Compose(
        #     [Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
        #      Resize(image_size + 20, image_size + 20, interpolation=2), RandomCrop(image_size, image_size)])

        # GT annotation
        GT = pd.read_csv(annot_path, header=None)

        self.datalist = []
        img_paths = glob.glob('{}/*.png'.format(dataset_path))
        with tqdm(enumerate(sorted(img_paths)), disable=True) as t:
            for wj, img_path in t:
                head, tail = os.path.split(img_path)
                img_id = tail  # Get image_id

                # check if it belongs to train/val set
                set = GT.loc[GT[0] == img_id][3]
                label = GT.loc[GT[0] == img_id][1]

                # Add only train/test to the corresponding set
                if set.iloc[0] == 'test':
                    if label.iloc[0] == 'HP':
                        cls_id = 0
                        self.num_neg += 1
                    else:
                        cls_id = 1  # SSA
                        self.num_pos += 1
                    self.datalist.append((img_path, cls_id))
                else:
                    continue



    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        image = Image.open(self.datalist[index][0]).convert('RGB')

        # label assignment
        target = int(self.datalist[index][1])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
    def get_stats(self):
        return self.num_pos,self.num_neg


#
if __name__ == "__main__":
    # imgs = make_dataset("/mnt/data/BCI/train/")
    # train_dataset = PatchCamelyon(
    #     '/mnt/data/patch_cam/pcamv1/',
    #     mode='train',
    #     augment=True
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=512, shuffle=False,
    #     num_workers=16, pin_memory=True)
    # training_loader_iter = iter(train_loader)
    # for i in range(3):
    #     x, y = next(training_loader_iter)
    #
    # for i, (images, target) in enumerate(train_loader):
    #     #print("\nBatch = " + str(batch_idx))
    #     #X = batch['gt_image']  # [3,7]
    #     print(i)
    #     break
    pass

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=16, shuffle=False,
    #     num_workers=8, pin_memory=True)
    #
    # from types import ModuleType
    # import inspect

    # for attribute in dir(train_loader):
    #     attribute_value = getattr(train_loader, attribute)
    #     #print(f'{attribute=}, {type(attribute_value)=}\n')
    #     if isinstance(attribute_value, ModuleType) or inspect.ismodule(attribute_value) or type(attribute_value) is type(inspect):
    #         print(attribute_value)
    # print('')

    # for i, (image, target) in enumerate(train_loader):
    #     #print("\nBatch = " + str(batch_idx))
    #     #X = batch['gt_image']  # [3,7]
    #     print(image.shape)
    #     print(target.shape)
    #     break

