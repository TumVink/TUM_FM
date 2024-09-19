import os
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import pathlib
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import _decompress, download_file_from_google_drive, verify_str_arg

from PIL import Image
from collections import OrderedDict


class PatchCamelyon(VisionDataset):
    _FILES = {
        "train": {
            "images": (
                "camelyonpatch_level_2_split_train_x.h5",  # Data file name
                "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",  # Google Drive ID
                "1571f514728f59376b705fc836ff4b63",  # md5 hash
            ),
            "targets": (
                "camelyonpatch_level_2_split_train_y.h5",
                "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
                "35c2d7259d906cfc8143347bb8e05be7",
            ),
        },
        "test": {
            "images": (
                "camelyonpatch_level_2_split_test_x.h5",
                "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
                "d8c2d60d490dbd479f8199bdfa0cf6ec",
            ),
            "targets": (
                "camelyonpatch_level_2_split_test_y.h5",
                "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
                "60a7035772fbdb7f34eb86d4420cf66a",
            ),
        },
        "val": {
            "images": (
                "camelyonpatch_level_2_split_valid_x.h5",
                "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
                "d5b63470df7cfa627aeec8b9dc0c066e",
            ),
            "targets": (
                "camelyonpatch_level_2_split_valid_y.h5",
                "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
                "2b85f58b927af9964a4c15b8f7e8f179",
            ),
        },
    }

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            atten_map: bool = False,
    ):
        # try:
        #     import h5py
        #
        #     self.h5py = h5py
        # except ImportError:
        #     raise RuntimeError(
        #         "h5py is not found. This dataset needs to have h5py installed: please run pip install h5py"
        #     )

        self._split = verify_str_arg(split, "split", ("train", "test", "val"))

        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root)
        self.cache_img = OrderedDict()
        self.cache_tgt = OrderedDict()
        self.max_cache_length = 4
        self.atten_map = atten_map

        # if download:
        #     self._download()

        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

    def __len__(self) -> int:
        images_file = self._FILES[self._split]["images"][0]
        with h5py.File(self._base_folder / images_file) as images_data:
            return images_data["x"].shape[0]



    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        images_file = self._FILES[self._split]["images"][0]
        if images_file in self.cache_img:
            image = Image.fromarray(self.cache_img[images_file][idx]).convert("RGB")
        else:
            if len(self.cache_img) > self.max_cache_length:
                # dictionay remove the first added item
                self.cache.popitem(last=False)
            with h5py.File(self._base_folder / images_file) as images_data:
                self.cache_img[images_file] = images_data["x"][:]
                image = Image.fromarray(images_data["x"][idx]).convert("RGB")

        targets_file = self._FILES[self._split]["targets"][0]
        if targets_file in self.cache_tgt:
            target = int(self.cache_tgt[targets_file][idx])
        else:
            if len(self.cache_tgt) > self.max_cache_length:  # shape is [num_images, 1, 1, 1]
                # dictionay remove the first added item
                self.cache.popitem(last=False)
            with h5py.File(self._base_folder / targets_file) as targets_data:
                self.cache_tgt[targets_file] = targets_data["y"][:,0,0,0,]
                target = int(targets_data["y"][idx, 0, 0, 0])
        if self.transform:
            img = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        if self.atten_map:
            transform_before_norm = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor()
                ]
            )
            target = transform_before_norm(image)


        return img, target


    # def _check_exists(self) -> bool:
    #     images_file = self._FILES[self._split]["images"][0]
    #     targets_file = self._FILES[self._split]["targets"][0]
    #     return all(self._base_folder.joinpath(h5_file).exists() for h5_file in (images_file, targets_file))


    # def _download(self) -> None:
    #     if self._check_exists():
    #         return
    #
    #     for file_name, file_id, md5 in self._FILES[self._split].values():
    #         archive_name = file_name + ".gz"
    #         download_file_from_google_drive(file_id, str(self._base_folder), filename=archive_name, md5=md5)
    #         _decompress(str(self._base_folder / archive_name))

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
    train_dataset = PatchCamelyon(root = '/mnt/data/patch_cam/',split='train',transform=transforms.Compose([
                                                 transforms.ColorJitter(brightness=.5, saturation=.25, hue=.1, contrast=.5),
                                                 transforms.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
                                                 transforms.RandomHorizontalFlip(.5),
                                                 transforms.RandomVerticalFlip(.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=path_default_mean,
                                                                      std=path_default_std)
                                                ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False,
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

