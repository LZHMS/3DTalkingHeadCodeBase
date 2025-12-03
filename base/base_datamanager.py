import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from tabulate import tabulate
from .base_dataset import build_dataset
from torch.utils.data import Dataset as TorchDataset

import logging
logger: logging.Logger

def build_data_loader(
    cfg,
    data_source=None,
    batch_size=64,
    shuffle=True,
    is_train=True,
    dataset_wrapper=None,
    infinite=False
):
    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Create dataset
    dataset = dataset_wrapper(cfg, data_source, is_train=is_train)
    
    # Determine if distributed training is enabled
    is_distributed = cfg.ENV.DISTRIBUTED and dist.is_initialized()
    
    # Use DistributedSampler for distributed training
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
        # When using sampler, shuffle must be False
        shuffle = False
    
    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=(torch.cuda.is_available() and cfg.ENV.USE_CUDA),
        drop_last=is_train  # Drop last incomplete batch in training for consistency
    )
    assert len(data_loader) > 0

    # Wrap with infinite iterator if requested
    if infinite:
        data_loader = InfiniteDataLoader(data_loader, sampler)

    return data_loader


def InfiniteDataLoader(data_loader, sampler=None):
    """Create an infinite iterator that loops over the data loader."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        for data in data_loader:
            yield data
        epoch += 1

class DataManager:

    def __init__(
        self,
        cfg,
        dataset_wrapper=None,
        infinite_train=False     # for iterate the dataloader when using iterator
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            shuffle=True,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
            infinite=infinite_train
        )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.show_dataset_summary(cfg)

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME

        table = []
        table.append(["Dataset", dataset_name])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        logger.info(f"Dataset summary:\n{tabulate(table)}")
    
    def data_analysis(self):
        pass


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, is_train=False):
        self.data_source = data_source
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {"index": idx}
        output.update(item.to_dict())
        
        return output