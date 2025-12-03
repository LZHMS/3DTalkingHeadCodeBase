"""
MNIST Dataset for Handwritten Digit Recognition
"""
import os
from torchvision import datasets, transforms

from base.base_dataset import Datum, DatasetBase, DATASET_REGISTRY
from base.base_datamanager import DataManager, DatasetWrapper

import logging
logger: logging.Logger


@DATASET_REGISTRY.register()
class MNIST(DatasetBase):
    """MNIST dataset for handwritten digit recognition
    """
    
    def __init__(self, cfg):
        # Data config and path
        print(cfg.ROOT)
        root = os.path.abspath(os.path.expanduser(cfg.ROOT))
        print(root)
        self.dataset_dir = os.path.join(root, cfg.NAME)
        print(self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)   # Create directory if not exists
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Load MNIST train and test datasets (will download if not exists)
        try:
            train_dataset = datasets.MNIST(
                root=self.dataset_dir,
                train=True,
                download=True,
                transform=self.transform
            )
            test_dataset = datasets.MNIST(
                root=self.dataset_dir,
                train=False,
                download=True,
                transform=self.transform
            )
        except Exception as e:
            raise f"Error loading MNIST dataset: {e}"
        
        # Convert to Datum objects
        train_data = []
        for idx, (image, label) in enumerate(train_dataset):
            train_data.append(
                Datum(name=f"train_{idx}", image=image, label=label)
            )
        
        # Process test data
        test_data = []
        for idx, (image, label) in enumerate(test_dataset):
            test_data.append(
                Datum(name=f"test_{idx}", image=image, label=label)
            )
        
        # Initialize base class (validation split will be handled by base class)
        super().__init__(train=train_data, val=None, test=test_data)
        
        # Auto-split validation set if needed: 0 < VAL_PERCENT < 1
        self.split_train_val(cfg.VAL_PERCENT, 42)


class MNISTDM(DataManager):
    """DataManager for MNIST dataset"""
    
    def __init__(self, cfg, dataset_wrapper=None, infinite_train=False):
        if dataset_wrapper is None:
            dataset_wrapper = MNISTWrapper

        super().__init__(cfg, dataset_wrapper, infinite_train)


class MNISTWrapper(DatasetWrapper):
    """Dataset wrapper for MNIST"""
    
    def __init__(self, cfg, data_source, is_train=False):
        super().__init__(cfg, data_source, is_train)
    
    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        output = {
            "index": idx,
            "name": item.name,
            "image": item.image,  # Image tensor
            "label": item.label  # Label (0-9)
        }
        
        return output

