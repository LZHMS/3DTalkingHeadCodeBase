import torch
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

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, is_train=is_train),
        batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=(torch.cuda.is_available() and cfg.ENV.USE_CUDA)
    )
    assert len(data_loader) > 0

    # Wrap with infinite iterator if requested
    if infinite:
        data_loader = InfiniteDataLoader(data_loader)

    return data_loader


def InfiniteDataLoader(data_loader):
    """Create an infinite iterator that loops over the data loader."""
    while True:
        for data in data_loader:
            yield data


class DataManager:

    def __init__(
        self,
        cfg,
        dataset_wrapper=None,
        infinite_train=False
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