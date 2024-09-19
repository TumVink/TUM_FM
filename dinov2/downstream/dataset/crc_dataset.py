import os
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.datasets import VisionDataset
import pathlib
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import _decompress, download_file_from_google_drive, verify_str_arg

from PIL import Image


# Define a dataset class
class CRC_Dataset(VisionDataset):
    def __init__(self, root_dir,split='train',transform=None):
        self.root_dir = root_dir
        if split=='train':
            self.root_dir = self.root_dir + 'NCT-CRC-HE-100K-NONORM/'
        else:
            self.root_dir = self.root_dir + 'CRC-VAL-HE-7K/'
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

        for label in os.listdir(self.root_dir):
            for img_file in os.listdir(os.path.join(self.root_dir, label)):
                self.image_paths.append(os.path.join(self.root_dir, label, img_file))
                self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return image, label

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
    path_default_mean = [0.70322989, 0.53606487, 0.66096631]
    path_default_std = [0.21716536, 0.26081574, 0.20723464]
    train_dataset = CRC_Dataset(root = '/mnt/data/crc/',split='train',transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=path_default_mean,
                                                                      std=path_default_std)
                                                ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=False,
        num_workers=16, pin_memory=True)

    from types import ModuleType
    import inspect

    # for attribute in dir(train_loader):
    #     attribute_value = getattr(train_loader, attribute)
    #     #print(f'{attribute=}, {type(attribute_value)=}\n')
    #     if isinstance(attribute_value, ModuleType) or inspect.ismodule(attribute_value) or type(attribute_value) is type(inspect):
    #         print(attribute_value)
    # print('')

    for i, (image, target) in enumerate(train_loader):
        #print("\nBatch = " + str(batch_idx))
        #X = batch['gt_image']  # [3,7]
        print(image.shape)
        print(target.shape)
        break

