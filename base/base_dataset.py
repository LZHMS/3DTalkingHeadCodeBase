import os
import os.path as osp
import gdown
import tarfile
import zipfile
import numpy as np

from utils.registry import Registry
from utils.tools import check_availability

import logging
logger: logging.Logger

DATASET_REGISTRY = Registry("DATASET")

def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.ENV.VERBOSE:
        logger.info("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg.DATASET)

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, name="", image=None, label=None, audio=None,
                 vertices=None, template=None, coefficients=None, impath=None):
        self._name = name
        self._impath = impath
        self._image = image
        self._label = label
        self._audio = audio
        self._vertices = vertices
        self._template = template
        self._coefficients = coefficients

    @property
    def name(self):
        return self._name
    
    @property
    def impath(self):
        return self._impath
    
    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    @property
    def audio(self):
        return self._audio

    @property
    def vertices(self):
        return self._vertices

    @property
    def template(self):
        return self._template
    
    @property
    def coefficients(self):
        return self._coefficients
    
    def to_dict(self, skip_none: bool = True):
        """
        Convert all non-private properties to a dictionary.
        
        Args:
            skip_none (bool): If True, skip attributes with None values.
        """
        out = {}
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            value = getattr(self.__class__, attr_name, None)
            if isinstance(value, property):
                val = getattr(self, attr_name)
                if skip_none and val is None:
                    continue
                out[attr_name] = val
        return out


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train=None, val=None, test=None):
        self._train = train  # labeled training data
        self._val = val  # validation data (optional)
        self._test = test  # test data

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test
    
    def split_train_val(self, val_percent, seed=42):
        """Split training data into train and validation sets.
        
        This is a utility method that can be called by dataset classes
        to automatically create a validation set from training data.
        
        Args:
            val_percent (float): Percentage of training data to use for validation (0.0 to 1.0)
            seed (int): Random seed for reproducible split
        """
        if self._train is None or len(self._train) == 0:
            logger.warning("No training data to split for validation")
            return
        
        if val_percent <= 0 or val_percent >= 1:
            logger.info(f"Skip automatically spliting val dataset.")
            return
        
        # Already has validation set, skip
        if self._val is not None and len(self._val) > 0:
            logger.info("Validation set already exists, skipping auto-split")
            return

        np.random.seed(seed)
        total_size = len(self._train)
        val_size = int(total_size * val_percent)
        
        if val_size == 0:
            logger.warning(f"Validation size is 0 (total: {total_size}, percent: {val_percent})")
            return
        
        # Random shuffle indices
        indices = np.random.permutation(total_size)
        
        # Split indices
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Save original train data before reassigning
        original_train = self._train
        
        # Create new splits
        self._train = [original_train[i] for i in train_indices]
        self._val = [original_train[i] for i in val_indices]
        logger.info(f"Split complete: {len(self._train)} train, {len(self._val)} val")

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))