# for the slides dataset patched by CLAM create_patches_fp.py
# With ['coords'] in the h5 file, we can read the patches from the h5 file
# by Jingsong Liu, 09/24/2024


import os
#import h5py
import h5pickle as h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.datasets import VisionDataset
import pathlib
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import _decompress, download_file_from_google_drive, verify_str_arg
from dinov2.ttt.stain_norm.models import stainNorm_Reinhard#, stainNorm_Macenko

from PIL import Image
import pandas as pd
import openslide
from collections import OrderedDict

def statistics(root_dir):
    for split in ['train','val','test']:
        if split =='train':
            split_dir = root_dir + 'split/train_slides.csv'
        elif split == 'val':
            split_dir = root_dir + 'split/valid_slides.csv'
        else:
            split_dir = root_dir + 'split/test_slides.csv'
        h5_path = '/mnt/nfs03-R6/CAMELYON17/additional_features/clam_small/patches/'
        data_pd = pd.read_csv(split_dir)
        num_patches_ls = []
        #read the 'patient' column
        slides_ls = data_pd['patient'].values.tolist()
        for slide_id in slides_ls: #slide_id = patient_024_node_2.tif
            slide_id = slide_id.split('.')[0] #patient_024_node_2
            slide_h5_path = h5_path + slide_id + '.h5'
            h5_file = h5py.File(slide_h5_path, 'r')
            num_patches = h5_file['coords'].shape[0]
            num_patches_ls.append(num_patches)
            h5_file.close()
        assert len(slides_ls) == len(num_patches_ls)
        # write num_patches_ls to a new column in the data_pd csv file
        data_pd['num_patches'] = num_patches_ls
        data_pd.to_csv(split_dir, index=False)
        print('total number of patches in ' + split + ' set: ' + str(sum(num_patches_ls)))

def check_cam16_magnification(root_dir):
    #check the magnification of the slides in the cam16 dataset
    slides_dir = os.path.join(root_dir,'images')
    slides_ls = os.listdir(slides_dir)
    mag_list = []
    level_list = []
    for slide_id in slides_ls: #slide_id = patient_024_node_2.tif
        slide = openslide.OpenSlide(os.path.join(slides_dir,slide_id))
        print(slide.level_downsamples)
        #mag_list.append(slide.properties['openslide.objective-power'])
        #level_list.append(slide.level_dimensions)
        #break
    # print(mag_list)
    # print(level_list)



# Define a dataset class
class Slides_Dataset(VisionDataset):
    #read from 1.tif file + 2. coords in h5 file
    def __init__(self, root_dir,split='train',transform=None):
        #self.root_dir = root_dir
        if split=='train':
            self.root_dir = root_dir + 'split/train_slides.csv'
        elif split == 'val':
            self.root_dir = root_dir + 'split/valid_slides.csv'
        else:
            self.root_dir = root_dir + 'split/test_slides.csv'
        self.data_pd = pd.read_csv(self.root_dir)
        self.img_dir = root_dir + 'images/'
        self.num_ls = self.data_pd['num_patches'].to_numpy()
        self.length = self.num_ls.sum()
        self.accumulate_ls = np.add.accumulate(self.num_ls)
        self.slide_id_ls = self.data_pd['patient'].values.tolist()
        self.transform = transform
        self.class_to_idx = {'itc': 0, 'negative': 1, 'macro': 2, 'micro': 3}
        if root_dir.contains('CAMELYON'):
            self.h5_path = root_dir+'/additional_features/clam_small/patches/'
        self.cache = OrderedDict()
        self.max_cache_length = 4


    def __len__(self):
        return sum(self.num_patches_ls)

    def __getitem__(self, idx):
        bag_candidate_idx = np.argwhere(self.accumulate_ls > idx).min()
        if bag_candidate_idx > 0:
            patch_local_idx = idx - self.accumulate_ls[bag_candidate_idx - 1]
        else:
            patch_local_idx = idx
        slide_id = self.slide_id_ls[bag_candidate_idx]

        with h5py.File(self.img_dir+slide_id, "r") as f:
            coord = f['coors'][patch_local_idx]

        if slide_id in self.cache:
            slide = self.cache[slide_id]
            patch = slide.crop((coord[0], coord[1], coord[0] + 512, coord[1] + 512))
        else:
            if len(self.cache) > self.max_cache_length:
                self.cache.popitem(last=False)
            slide = Image.open(self.img_dir, slide_id)
            self.cache[slide_id] = slide
            patch = slide.crop((coord[0], coord[1], coord[0] + 512, coord[1] + 512))


        if self.transform:
            patch = self.transform(patch)

        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return patch, idx

class Multi_Slides(VisionDataset):
    #read from 1.tif file + 2. coords in h5 file
    def __init__(self, root_dir,slides_df,split='train',transform=None):
        #self.root_dir = root_dir
        self.data_pd = pd.read_csv(slides_df)
        #selfect the first 10 rows
        #self.data_pd = self.data_pd.iloc[:10, :]
        if 'CAMELYON' in root_dir:
            key_word = "tbs"
        else:
            key_word = "processed"
        mask = self.data_pd['status'] == key_word
        self.data_pd = self.data_pd[mask]
        self.num_ls = self.data_pd['process'].to_numpy()
        self.length = self.num_ls.sum()
        self.accumulate_ls = np.add.accumulate(self.num_ls)
        self.slide_id_ls = self.data_pd['slide_id'].values.tolist()
        self.transform = transform
        self.class_to_idx = {'itc': 0, 'negative': 1, 'macro': 2, 'micro': 3}
        #if root_dir.contains('CAMELYON'):
        self.h5_path = os.path.join(root_dir,'patches')
        self.cache = OrderedDict()
        self.max_cache_length = 8


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        bag_candidate_idx = np.argwhere(self.accumulate_ls > idx).min()
        if bag_candidate_idx > 0:
            patch_local_idx = idx - self.accumulate_ls[bag_candidate_idx - 1]
        else:
            patch_local_idx = idx
        slide_id = self.slide_id_ls[bag_candidate_idx]
        slide_dir = os.path.join(self.h5_path,os.path.basename(slide_id).split('.')[0]+'.h5')

        # first search imgs in cache
        if slide_dir in self.cache:
            image = Image.fromarray(self.cache[slide_dir][patch_local_idx], mode="RGB")
        else:
            with h5py.File(slide_dir, "r") as f:
                if len(self.cache) > self.max_cache_length:
                    # dictionay remove the first added item
                    self.cache.popitem(last=False)
                self.cache[slide_dir] = f['imgs'][:]
                image = Image.fromarray(f['imgs'][patch_local_idx], mode="RGB")


        if self.transform:
            patch = self.transform(image)

        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return patch, idx


# Define a dataset class
class Single_slide_dataset_coords(VisionDataset):
    def __init__(self,root_dir, slide_id,num_patches,split='train',transform=None):
        #self.root_dir = root_dir
        # if split=='train':
        #     self.root_dir = root_dir + 'split/train_slides.csv'
        # elif split == 'val':
        #     self.root_dir = root_dir + 'split/valid_slides.csv'
        # else:
        #     self.root_dir = root_dir + 'split/test_slides.csv'
        #self.data_pd = pd.read_csv(self.root_dir)
        self.img_dir = root_dir + 'images/'
        #self.num_ls = self.data_pd['num_patches'].to_numpy()
        #self.length = self.num_ls.sum()
        #self.accumulate_ls = np.add.accumulate(self.num_ls)
        #self.slide_id_ls = self.data_pd['patient'].values.tolist()
        self.transform = transform
        self.class_to_idx = {'itc': 0, 'negative': 1, 'macro': 2, 'micro': 3}
        self.slide_id = slide_id.split('.')[0]+'.h5'
        if 'CAMELYON' in root_dir:
            self.h5_path = root_dir+'/additional_features/clam_small/patches/'+self.slide_id
        self.h5 = h5py.File(self.h5_path, "r")
            #self.coord = f['coords']
        #self.slide = Image.open(os.path.join(self.img_dir, slide_id))
        self.patch_level = self.h5['coords'].attrs['patch_level']
        self.patch_size = self.h5['coords'].attrs['patch_size']
        self.wsi = openslide.OpenSlide(os.path.join(self.img_dir, slide_id))
        self.len = num_patches
        # self.cache = OrderedDict()
        # self.max_cache_length = 16


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        coord = self.h5['coords'][idx]
        # coord = coords[idx]
        patch = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        if self.transform:
            patch = self.transform(patch)

        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return patch


class Single_slides_img(VisionDataset):
    #patched tile with RGB channels into h5 files, input is stil single slide
    def __init__(self,root_dir, slide_id,num_patches,split='train',transform=None):

        self.transform = transform
        self.class_to_idx = None
        self.slide_id = slide_id

        self.h5_path = os.path.join(root_dir,'patches',self.slide_id)

        self.h5 = h5py.File(self.h5_path, "r")
            #self.coord = f['coords']
        #self.slide = Image.open(os.path.join(self.img_dir, slide_id))
        # self.patch_level = self.h5['coords'].attrs['patch_level']
        # self.patch_size = self.h5['coords'].attrs['patch_size']
        #self.wsi = openslide.OpenSlide(os.path.join(self.img_dir, slide_id))
        self.len = num_patches
        #
        self.len = len(self.h5['imgs'])


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        patch = Image.fromarray(self.h5['imgs'][idx], mode="RGB")
        if self.transform:
            patch = self.transform(patch)
        coords = self.h5['coords'][idx]
        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return patch,idx, coords

class Single_slides_img_norm(VisionDataset):
    #patched tile with RGB channels into h5 files, input is stil single slide
    def __init__(self,root_dir, slide_id,num_patches,split='train',transform=None,stain_norm='reinhard',target_means=None,target_stds=None):

        self.transform = transform
        self.class_to_idx = None
        self.slide_id = slide_id

        self.h5_path = os.path.join(root_dir,'patches',self.slide_id)

        self.h5 = h5py.File(self.h5_path, "r")
            #self.coord = f['coords']
        self.len = num_patches
        #
        self.len = len(self.h5['imgs'])
        self.stain_normalizer = stainNorm_Reinhard.Normalizer() if stain_norm == 'reinhard' else stainNorm_Macenko.Normalizer()
        # Yale Her2 statistic
        # self.stain_normalizer.target_means = (74.63404643764446, 22.025532559201558, -9.475832381887503)
        # self.stain_normalizer.target_stds = (19.14838739629965, 12.540998432005603, 9.982434700221374)
        # TCGA survival statistic
        self.stain_normalizer.target_means = target_means
        self.stain_normalizer.target_stds = target_stds

    def __len__(self):
        return self.len

    def __getitem__(self, idx):


        #stain_norm
        if self.stain_normalizer is not None:
            cv2_normalized_image = self.stain_normalizer.transform(self.h5['imgs'][idx])
            patch = Image.fromarray(cv2_normalized_image, mode="RGB")


        if self.transform:
            patch = self.transform(patch)
        coords = self.h5['coords'][idx]
        # Print the shape of the image tensor if you want to confirm
        # print(f'Shape of image tensor (after resizing): {image.size()}')

        return patch,idx, coords

#
if __name__ == "__main__":
    check_cam16_magnification('/mnt/nfs03-R6/CAMELYON16/')
    #statistics('/mnt/nfs03-R6/CAMELYON17/')
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
    # path_default_mean = [0.70322989, 0.53606487, 0.66096631]
    # path_default_std = [0.21716536, 0.26081574, 0.20723464]
    # train_dataset = CRC_Dataset(root = '/mnt/data/crc/',split='train',transform=transforms.Compose([
    #                                              transforms.ToTensor(),
    #                                              transforms.Normalize(mean=path_default_mean,
    #                                                                   std=path_default_std)
    #                                             ]))
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=16, shuffle=False,
    #     num_workers=16, pin_memory=True)
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

