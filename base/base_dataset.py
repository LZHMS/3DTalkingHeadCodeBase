import os
import os.path as osp
import gdown
import tarfile
import zipfile

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

    def __init__(self, name="", audio=None, vertices=None, template=None, coefficients=None, impath=None):
        # for fpath in [impath, audio_path, vertices_path, template_file]:
        #     assert isinstance(fpath, str)
        #     assert check_isfile(fpath)
        self._name = name
        self._impath = impath
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