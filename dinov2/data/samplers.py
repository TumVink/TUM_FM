# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
from typing import Any, Optional
import warnings
from typing import TypeVar, Iterator

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

import dinov2.distributed as distributed
from dinov2.data.datasets.CustomImageDataset import TUM_slides

T_co = TypeVar('T_co', covariant=True)

class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


class TUM_DistributedSampler(Sampler[T_co]):
    """Foundation Model Distributed Sampler for TUM slides. Splits the training files equally between different ranks
    and assigns patches from the files to the ranks. Makes sure that the cache from the dataset
    is efficiently used, i.e. consecutive patches are loaded from the same file before moving to the next file.
        Taken from https://github.com/facebookresearch/dinov2/pull/461/files#diff-6550a8c01c5663760eecd4071a8431155a2ddf6ba14659413090ad2f9aa45921
    Args:
        dataset: Dataset used for sampling.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self, dataset: TUM_slides, shuffle: bool = False, seed: int = 0) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        # create file_ids
        num_files = len(self.dataset.slide_id_ls)
        print('num_files:'+str(num_files))
        if num_files % self.num_replicas != 0:
            # make sure to split files evenly
            num_files = (num_files // self.num_replicas + 1) * self.num_replicas
            print('after sample num_files:' + str(num_files))
        # make sure file indices are circular
        self.file_ids = [i % len(self.dataset.slide_id_ls) for i in range(num_files)]
        if self.shuffle:
            # randomize file_ids
            rs = np.random.RandomState(seed)
            rs.shuffle(self.file_ids)
            # init random generator for shuffling indices
            self.rs = np.random.RandomState(seed + self.rank)
        # partition the files between different ranks
        file_ids = np.array_split(self.file_ids, self.num_replicas)[self.rank]
        # convert file ids to 2D patch indices array
        # self.indices = []
        # for file_id in file_ids:
        #     start_idx = file_id * self.dataset.internal_patch_count
        #     end_idx = start_idx + self.dataset.internal_patch_count
        #     self.indices.append(list(range(start_idx, end_idx)))

        sum_indices = np.cumsum(self.dataset.num_ls[file_ids])  # it should start from 0, then N1
        end_indices = sum_indices - 1

        start_indices = np.insert(sum_indices, 0, 0)
        start_indices = start_indices[:-1]

        assert len(start_indices) == len(end_indices)
        self.indices = []
        for i in range(len(start_indices)):
            self.indices.append(list(range(start_indices[i], end_indices[i] + 1)))

        # compute number of patches
        self.num_samples =self.dataset.num_ls[file_ids].sum() #len(file_ids) * self.dataset.internal_patch_count

    def __iter__(self) -> Iterator[T_co]:
        yield from self._iterator()

    def _iterator(self):
        if self.shuffle:
            self._shuffle_indices()
        index_iterator = itertools.chain(*self.indices)
        i = 0
        while True:
            yield from index_iterator
            i += 1
            # reset the index and reshuffle if necessary
            if i >= self.num_samples:
                i = 0
                if self.shuffle:
                    # reshuffle the indices
                    self._shuffle_indices()

    def _shuffle_indices(self):
        """We must shuffle within the individual rows to ensure data locality and efficient cache usage."""
        # shuffle the rows
        self.rs.shuffle(self.indices)
        # shuffle withing the individual rows
        for ind in self.indices:
            self.rs.shuffle(ind)