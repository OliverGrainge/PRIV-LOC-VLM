import os
from glob import glob

from dataloaders.gptgeochat import GPTGeoChat
from dataloaders.mp16 import MP16
from dataloaders.standarddataset import StandardDataset
from dataloaders.osv5 import OSV5Dataset


def load_dataset(dataset_name, *args, **kwargs):
    DATASET_CLASSES = {
        "im2gps": StandardDataset,
        "im2gps3k": StandardDataset,
        "yfcc4k": StandardDataset,
        "mp16": MP16,
        "gptgeochat": GPTGeoChat,
        "osv5": OSV5Dataset,
    }

    if dataset_name not in DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Make sure it's registered in DATASET_CLASSES."
        )

    dataset_class = DATASET_CLASSES[dataset_name]
    return dataset_class(dataset_name, *args, **kwargs)


def collate_fn(batch):
    """Custom collate function to prevent PyTorch from converting PIL images into tensors."""
    images = [item["image"] for item in batch]
    paths = [item["path"] for item in batch]
    return {"images": images, "paths": paths}
