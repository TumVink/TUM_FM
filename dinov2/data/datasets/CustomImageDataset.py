# this file was added completely new

import os, io
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
# from sklearn.model_selection import train_test_split
from .decoders import ImageDataDecoder
from torchvision.transforms.functional import to_pil_image
import random
import numpy as np
from PIL import Image
import h5py
from collections import OrderedDict
import time
from cucim import CuImage
from collections import OrderedDict
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import tarfile
import math
from pathlib import Path


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

    def __getitem__(self, idx):
        # count the time for loading data and transform the data
        time_start = time.time()


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
        time_loading = time.time() - time_start
        #print(image.size)
        if self.transform:  # /mnt/nfs03-R6/TCGA/clam/patches/TCGA-02-0001-01Z-00-DX1.h5
            image = self.transform(image)
        time_transformed = time.time() - time_start - time_loading
        # print('time_loading:', time_loading)
        # print('time_transformed:', time_transformed)
        return image, None

class SlideDatasetCucim(Dataset):
    def __init__(self, h5file, slidefile, wsi = None,transform=None):
        self.h5file = h5file
        self.tforms = []
        self.tforms.append(transforms.Resize(224))
        self.tforms.append(transforms.ToTensor())
        self.tforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.tforms = transforms.Compose(self.tforms)
        self.slidefile = slidefile
        if wsi is None:
            self.wsi = CuImage(self.slidefile)
        else:
            self.wsi = wsi

        with h5py.File(h5file, 'r') as f:
            self.dset = f['coords'][:]
            if 'patch_level' in f['coords'].attrs:
                self.patch_level = f['coords'].attrs['patch_level']
            else:
                self.patch_level = 0
            #self.patch_level = f['coords'].attrs['patch_level']
            if self.patch_level == 0:
                self.patch_size = f['coords'].attrs['patch_size_level0']
            else:
                self.patch_size = f['coords'].attrs['patch_size']
            print('Patch_size:', self.patch_size)
            print('Patch_level:', self.patch_level)


    def __len__(self):
        return len(self.dset)


    def __getitem__(self, idx):
        coord = self.dset[idx]
        x, y = coord
        img = self.wsi.read_region(location=(x, y), size=(self.patch_size, self.patch_size), level=0)
        img = np.asarray(img, dtype=np.uint8)
        img = to_pil_image(img)
        #img = self.tforms(img)

        return img

class PathDataset(Dataset):
    def __init__(self, csv='/mnt/nfs01-R0/TUM_breast_cancer/clam_20/process_list_autogen.csv', patch_dir='/mnt/nfs01-R0/TUM_breast_cancer/clam_20/coor/',
                 slide_dir='/mnt/nfs01-R0/TUM_breast_cancer/svs/',transform=None, target_transform=None):
        # self.slide_ids = slide_ids
        #print('Slide_ids:', self.slide_ids)
        # self.patch_dir = patch_dir
        # self.slide_dir = slide_dir
        # with open(csv, "r") as f:
        #     slide_ids = [line.strip() for line in f]
        #     slide_ids = [slide_id.split(',')[0].split('.')[0].split('/')[-1] for slide_id in slide_ids]
        #     # slide_ids = [slide_id.split('.')[0] for slide_id in slide_ids]
        #     slide_ids = slide_ids[1:]
        # print(slide_ids[:10]
        self.df = pd.read_csv('/home/ge54xof/dino-tum/dinov2/data/datasets/public_dataset.csv')
        self.slides = self.df['slide'].tolist()
        self.coords = self.df['coor'].tolist()
        self.transform = transform

        self.cache = OrderedDict()
        self.cache_size = 50

        #self.cache = defaultdict(lambda: {'image': None, 'indices': None,'patch_level': None, 'patch_size': None})

    def __len__(self):
        return len(self.slides)*500

    def __getitem__(self, idx):
        slide_idx = int(idx // 500)
        patch_idx = int(idx % 500)
        slide_file = self.slides[slide_idx]
        patch_file = self.coords[slide_idx]
        #print(slide_id)
        # patch_file = os.path.join(self.patch_dir, slide_id + '.h5')
        # #patch_file = os.path.join(self.patch_dir, slide_id.split('/')[-1] + '.h5')
        # slide_file = os.path.join(self.slide_dir, slide_id + '.svs')
        #print(patch_file, slide_file)
        # patch_dataset = SlideDatasetCucim(patch_file, slide_file)
        # dataloader = DataLoader(patch_dataset, batch_size=500, shuffle=False, pin_memory=False)
        # for batch in dataloader:
        #     patches.append(batch)

        # Check if the slide is already cached
        if slide_id not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove the oldest item (FIFO)
            # Load the slide image if not cached
            wsi = CuImage(slide_file)
            with h5py.File(patch_file, 'r') as f:
                coords = f['coords'][:]
                indices = random.sample(range(len(coords)), 500)
                if 'patch_level' in f['coords'].attrs:
                    patch_level = f['coords'].attrs.get('patch_level', 0)
                    patch_size = f['coords'].attrs['patch_size']
                else:
                    #hard coding for yale dataset
                    patch_level=0
                    patch_size=256
                # patch_level = f['coords'].attrs.get('patch_level', 0)
                # patch_size = f['coords'].attrs['patch_size']

            self.cache[slide_id] = {
                'image': wsi,
                'coords': coords,
                'indices': indices,
                'patch_level': patch_level,
                'patch_size': patch_size,
            }
        self.cache.move_to_end(slide_id)

        # Use the cached indices to fetch the patch coordinates
        slide_cache = self.cache[slide_id]
        coord = slide_cache['coords'][slide_cache['indices'][patch_idx]]
        x, y = coord

        # Read the region from the slide image
        img = slide_cache['image'].read_region(location=(x, y), size=(slide_cache['patch_size'], slide_cache['patch_size']),
                                                        level=slide_cache['patch_level'])
        # resize to 256x256 if patch_size is not 256
        img = np.asarray(img, dtype=np.uint8)
        img = to_pil_image(img)
        if slide_cache['patch_size'] != 256:
            img = img.resize((256, 256), Image.LANCZOS)
        if self.transform:
            img = self.transform(img)

        return img, None

        # self.wsi = CuImage(slide_file)
        # with h5py.File(patch_file, 'r') as f:
        #     #randomly select 500 patches, if there are less than 500 patches, repeatly selected 500
        #     self.dset = f['coords'][:]
        #     index = random.sample(range(len(self.dset)), 500)
        #     self.dset = self.dset[index]
        #     if 'patch_level' in f['coords'].attrs:
        #         self.patch_level = f['coords'].attrs['patch_level']
        #     else:
        #         self.patch_level = 0
        #     self.patch_size = f['coords'].attrs['patch_size']
        # coord = self.dset[patch_idx]
        # x, y = coord
        # img = self.wsi.read_region(location=(x, y), size=(self.patch_size, self.patch_size), level=0)
        # img = np.asarray(img, dtype=np.uint8)
        # #print(img.shape)
        # img = to_pil_image(img)

        # if self.transform:  # /mnt/nfs03-R6/TCGA/clam/patches/TCGA-02-0001-01Z-00-DX1.h5
        #     img = self.transform(img)
        #
        # # instead, load the patches with coords from h5 file using cucim
        # #img is a dictionary with keys 'global_crops' and 'local_crops'
        # return img, None#, slide_id#.split('/')[-1]

from concurrent.futures import ThreadPoolExecutor
class PathDataset_mt(Dataset):
    def __init__(self, csv='/mnt/nfs01-R0/TUM_breast_cancer/clam_20/process_list_autogen.csv', patch_dir='/mnt/nfs01-R0/TUM_breast_cancer/clam_20/coor/',
                 slide_dir='/mnt/nfs01-R0/TUM_breast_cancer/svs/', transform=None, cache_size=50, max_workers=16,target_transform=None):
        self.df = pd.read_csv('/home/ge54xof/dino-tum/dinov2/data/datasets/public_dataset.csv')
        self.slides = self.df['slide'].tolist()
        self.coords = self.df['coor'].tolist()
        self.transform = transform

        self.transform = transform

        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def __len__(self):
        return len(self.slides) * 500

    def _load_slide_to_cache(self, slide_id, patch_file, slide_file):
        wsi = CuImage(slide_file)
        with h5py.File(patch_file, 'r') as f:
            coords = f['coords'][:]
            indices = random.choices(range(len(coords)), k=500)
            patch_level = f['coords'].attrs.get('patch_level', 0)
            patch_size = f['coords'].attrs.get('patch_size', 256)
        return {
            'image': wsi,
            'coords': coords,
            'indices': indices,
            'patch_level': patch_level,
            'patch_size': patch_size
        }

    def __getitem__(self, idx):
        slide_idx = idx // 500
        patch_idx = idx % 500
        slide_file = self.slides[slide_idx]
        patch_file = self.coords[slide_idx]

        # Load slide to cache if needed
        if slide_file not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[slide_file] = self._load_slide_to_cache(slide_file, patch_file, slide_file)
        self.cache.move_to_end(slide_file)

        slide_cache = self.cache[slide_file]
        coord = slide_cache['coords'][slide_cache['indices'][patch_idx]]
        x, y = coord

        def load_patch():
            img = slide_cache['image'].read_region(
                location=(x, y),
                size=(slide_cache['patch_size'], slide_cache['patch_size']),
                level=slide_cache['patch_level']
            )
            img = np.asarray(img, dtype=np.uint8)
            img = to_pil_image(img)
            if slide_cache['patch_size'] != 256:
                img = img.resize((256, 256), Image.LANCZOS)
            if self.transform:
                img = self.transform(img)
            return img

        # Run the patch loading in a thread
        future = self.thread_pool.submit(load_patch)
        img = future.result()

        return img, None


class TUMShardDataset(IterableDataset):
    def __init__(self, root, transform=None,target_transform=None, chunk_size=1000):
        self.shards = list(Path(root).glob("*.tar"))
        self.shards = [str(shard) for shard in self.shards]
        #self.shards = shards  # list of .tar files
        self.transform = transform
        self.chunk_size = chunk_size
        self.num_shards = len(self.shards)
        # self.rank = dist.get_rank() if dist.is_initialized() else 0
        # self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        random.Random(epoch).shuffle(self.shards)
        #self.local_shards = self._get_local_shards()

    def _get_local_shards(self):
        return self.shards[self.rank * self.partition_size:(self.rank + 1) * self.partition_size]

    def __len__(self):
        return self.num_shards

    def __iter__(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.partition_size = math.ceil(self.num_shards / world_size)
        self.local_shards = self._get_local_shards()

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        for shard_path in self.local_shards[worker_id::num_workers]:
            try:
                with tarfile.open(shard_path, 'r') as tar:
                    members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg'))]
                    random.shuffle(members)  # optional: shuffle images inside tar
                    for i in range(0, len(members), self.chunk_size):
                        chunk = members[i:i + self.chunk_size]
                        for member in chunk:
                            try:
                                f = tar.extractfile(member)
                                if f is None:
                                    continue
                                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                                if self.transform:
                                    img = self.transform(img)
                                yield img, None  # <- match your current TUM_slides output
                            except Exception:
                                continue
            except Exception as e:
                print(f"Error reading shard {shard_path}: {e}")







