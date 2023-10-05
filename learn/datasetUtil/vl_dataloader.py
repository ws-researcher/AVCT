"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import os.path as op
import torch
from torch.utils.data import DataLoader

from learn.datasetUtil.caption_tensorizer import build_tensorizer
from learn.datasetUtil.dataset import VisionLanguageTSVDataset
from utils.comm import get_world_size


def make_data_loader(args, tokenizer_text, is_train=True):
    tokenizer_multimodal = build_tensorizer(args, tokenizer_text, is_train)
    dataset = VisionLanguageTSVDataset(args, tokenizer_multimodal, is_train)
    a, b, c = dataset.__getitem__(10)
    if is_train:
        shuffle = True
        start_iter = 0
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    distributed = not get_world_size() == 1
    sampler = make_data_sampler(dataset, shuffle=shuffle, distributed=distributed, random_seed=args.seed)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)

    data_loader = DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, worker_init_fn=init_seeds,
    )
    return data_loader

class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle=True, distributed=True, random_seed=0):
    if distributed:
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=random_seed)
    else:
        pass

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def init_seeds(seed=88):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)