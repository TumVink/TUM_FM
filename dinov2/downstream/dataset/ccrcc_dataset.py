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

def make_dataset(data_dir = '/mnt/nfs03-R6/CCRCC_patch_cls/tissue_classification/'):
    # extract dataset info into a csv file
    # data_stucture: cancer/TCGA*.png or normal/TCGA*.png or stroma/TCGA*.png for training
    # data_stucture: cancer/FIMM*.png or normal/FIMM*.png or stroma/FIMM*.png for testing
    # return: a csv file with columns 'image path', 'label', 'split' with info
    #                          cancer/TCGA*.png, 1, train
    label_dict = {'cancer': 1, 'normal': 0, 'stroma': 2,'blood': 3,'empty': 4,'other': 5}
    split_dict = {'train': 'train', 'test': 'test'}
    data = []
    for label in label_dict.keys():
        img_paths = glob.glob('{}{}/*.png'.format(data_dir, label))
        for img_path in img_paths:
            if 'TCGA' in img_path:
                split = 'train'
            else:
                split = 'test'
            data.append([img_path, label_dict[label], split_dict[split]])
    df = pd.DataFrame(data, columns=['image path', 'label', 'split'])
    df.to_csv('/mnt/nfs03-R6/CCRCC_patch_cls/tissue_classification/CCRCC_patch_cls.csv', index=False)


class CCRCC_patch(VisionDataset):
    def __init__(
            self,
            root: str = '/mnt/nfs03-R6/CCRCC_patch_cls/tissue_classification/',
            split: str = "train", # only train and test
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,):

        """
        MHIST dataset class wrapper (train with augmentation)
        """

        #self.image_size = image_size

        self.split = split
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        #self.img_dir = self.root
        self.csv_file = os.path.join(self.root, 'CCRCC_patch_cls.csv')
        df = pd.read_csv(self.csv_file)
        mask = df['split'] == self.split
        self.imgs_ls = df['image path'][mask].values.tolist()
        self.labels_ls = df['label'][mask].values.tolist()


    def __len__(self):
        #images_file = self._FILES[self._split]["images"][0]
        return len(self.imgs_ls)

    def __getitem__(self, idx):


        img_path = self.imgs_ls[idx]
        target = self.labels_ls[idx]
        #read png image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target






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
    make_dataset()

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

