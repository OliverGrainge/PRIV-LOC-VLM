import json
from abc import ABC, abstractmethod

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from config import DATASETS


class BaseDataset(Dataset, ABC):
    def __init__(self, dataset_name, image_size=(512, 512)):
        """
        Args:
            metadata (str): path to dataset metadata (.csv)
            data_root (str): Path to the img dir.
            transform (callable, optional): Optional transform to be applied to images.
        """
        with open(DATASETS, "r") as f:
            datasets = json.load(f)

        assert (
            dataset_name in datasets.keys()
        ), f'{dataset_name} is not a valid dataset. Specify the \
            metadata and image paths in a JSON file and configure it in config under "DATASETS".'
        self._dataset = datasets[dataset_name]
        if self._dataset["meta"]:
            self.metadata_pth = self._dataset["meta"]
        self.data_root = self._dataset["images"]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
