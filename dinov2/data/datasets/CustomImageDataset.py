# this file was added completely new

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
# from sklearn.model_selection import train_test_split
from .decoders import ImageDataDecoder
import random
import numpy as np
from PIL import Image
import h5py
from collections import OrderedDict


def sample_patches_from_csv(total_batches=5):
    '''
        Sample_patches id from the first 5 patches
        num_batches: int
        return: df of patches for the first total_batches
    '''
    # read the csv file in dir /home/ge54xof/Foundation-Model-for-Pathology/data/, with name local_slides_df_bt_+'num_batches'+.csv

    local_slides_dfs = [
        os.path.join('/home/ge54xof/Foundation-Model-for-Pathology/data/local_slides_df_bt_' + str(nr_batch) + '.csv')
        for nr_batch in range(total_batches)]
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
            'status': ['tb trained'] * len(local_num_patches_ls),  # 'deleted', 'training', 'tb trained'
            'num_patches': local_num_patches_ls

        }
        # print('slide_length:'+str(len(local_slides_paths_ls)))
        # print('status:' + str(len(local_slides_df['status'])))
        # print('num_patches:'+str(len(local_num_patches_ls)))
        local_slides_df = pd.DataFrame(local_slides_df)

    return local_slides_df


# these functions were just created to check how the augmented images/crops look like
def save_image(tensor, name1, name2):
    # Convert the tensor to a NumPy array
    numpy_array = tensor.cpu().numpy()

    # Convert to uint8 and scale to [0, 255]
    numpy_array = (numpy_array * 255).astype('uint8')

    # Create an image from the NumPy array
    image = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Convert to HWC format

    # Concatenate name1 and name2 to form the file name
    file_name = f'{name1}_{name2}.png'

    # Specify the save path with the concatenated file name
    save_path = f'/home/ge54xof/dino-tum/dinov2/data/viz/{file_name}'

    # Save the image
    image.save(save_path)


def save_all(image_pil, name1,global_num,local_num):
    # save both global crops
    assert global_num <= len(image_pil['global_crops'])
    assert local_num <= len(image_pil['local_crops'])
    #print(image_pil['global_crops'].shape)
    # for i in range(len(image_pil['global_crops'])):
    #     print(i)
    save_image(image_pil['global_crops'][0], name1, 'global_crop_'+str(0))
    save_image(image_pil['global_crops'][1], name1, 'global_crop_' + str(1))
    #save_image(image_pil['global_crops'][1], name1, 'global_crop2')
    for i in range(local_num):
        save_image(image_pil['local_crops'][local_num], name1, 'local_crop_'+str(i))

    # save local crops
    # local_crops = image_pil['local_crops']
    # for i, image in enumerate(local_crops):
    #     save_image(image, name1, 'local_crop' + str(i))


class CustomImageDataset(Dataset):
    def __init__(
            self,
            split,
            root: str,
            extra: str,
            transform=None,
            target_transform=None,
            test_size=0.2,
            random_state=42):
        self.img_labels = pd.read_csv(extra)
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # split into train and test
        # self.train_data, self.test_data = train_test_split(
        #    self.img_labels, test_size=test_size, random_state=random_state)

    def __len__(self):
        return len(self.img_labels)
        # return len(self.train_data)

    def __getitem__(self, idx):
        try:
            img_path = self.img_labels.iloc[idx, 0]
            with open(img_path, mode="rb") as f:
                image_pil = f.read()
            image_pil = ImageDataDecoder(image_pil).decode()
        except Exception as e:
            # in case of error when reading image, just take a random different one
            random_index = random.randint(0, len(self) - 1)
            image = self.__getitem__(random_index)
            return image, None
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, None

    def get_test_item(self, idx):
        img_path = os.path.join(self.img_dir, self.test_data.iloc[idx, 0])
        image = read_image(img_path)
        image_pil = transforms.ToPILImage()(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil


class TUM_slides(Dataset):

    def __init__(
            self,
            split,
            root: str,
            extra: str,
            target_patch_size=None, patches_per_slide=500,
            transforms=None,
            transform=None,
            target_transform=None,
    ):
        self.img_dir = root
        self.target_patch_size = target_patch_size
        self.patches_per_slide = patches_per_slide
        self.transform = transform
        self.slide_ext = '.h5'
        # self.data_h5_path = "/mnt/data/"

        self.local_slides_df = pd.read_csv('/home/ge54xof/dino-tum/dinov2/data/datasets/TUM_100K.csv')

        self.num_ls = self.local_slides_df.num_patches.to_numpy()
        self.length = self.num_ls.sum()
        self.accumulate_ls = np.add.accumulate(self.num_ls)
        self.slide_id_ls = self.local_slides_df.slide_id.tolist()
        self.split = split
        self.target_transform = target_transform

        # self.slides_status.to_csv('slides_status.csv', index=False)
        # np.random.seed(0)
        self.cache = OrderedDict()
        self.max_cache_length = 16

    def __len__(self):
        return self.length  # todo

    # def __getitem__(self, idx):
    #
    #     # select the patch
    #     bag_candidate_idx = np.argwhere(self.accumulate_ls > idx).min()    #]
    #     if bag_candidate_idx>0:
    #         patch_local_idx = idx - self.accumulate_ls[bag_candidate_idx - 1]
    #     else:
    #         patch_local_idx = idx
    #
    #     slide_id = self.slide_id_ls[bag_candidate_idx]
    #     #print('Fking some error happens here with .loc()')
    #     slide_dir = slide_id
    #     print(slide_id)
    #     try:
    #         with h5py.File(slide_dir, "r") as f:
    #             print('try opening')
    #             # try:
    #             image = Image.fromarray(f['imgs'][patch_local_idx], mode="RGB")
    #     except Exception as e:
    #         print(e)
    #         # in case of error when reading image, just take a random different one
    #         random_index = random.randint(0, len(self) - 1)
    #         image = self.__getitem__(random_index)
    #         return image, None
    #     print('finished reading')
    #     if self.transform:
    #         image = self.transform(image)
    #     print('finished')
    #     return image,None

    def __getitem__(self, idx):
        # load patches more efficiently by adding patches to cache
        # select the patch
        bag_candidate_idx = np.argwhere(self.accumulate_ls > idx).min()  # ]
        if bag_candidate_idx > 0:
            patch_local_idx = idx - self.accumulate_ls[bag_candidate_idx - 1]
        else:
            patch_local_idx = idx
        slide_id = self.slide_id_ls[bag_candidate_idx]
        slide_dir = slide_id

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

        if self.transform:  # /mnt/nfs03-R6/TCGA/clam/patches/TCGA-02-0001-01Z-00-DX1.h5
            image = self.transform(image)

        return image, None





