from typing import List

import pytest
from PIL import Image
from torch.utils.data import DataLoader

from dataloaders import ALL_DATASET_NAMES
from dataloaders.helper import collate_fn, load_dataset


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataloader_instantiation(dataset_name):
    dataset = load_dataset(dataset_name)
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, collate_fn=collate_fn, num_workers=0
    )
    assert dataloader is not None


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataloader_format(dataset_name):
    dataset = load_dataset(dataset_name)
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, collate_fn=collate_fn, num_workers=0
    )
    batch = next(iter(dataloader))
    assert batch is not None
    assert len(batch["images"]) == 2, type(batch["images"])
    assert len(batch["paths"]) == 2, type(batch["paths"])


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataloader_types(dataset_name):
    dataset = load_dataset(dataset_name)
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, collate_fn=collate_fn, num_workers=0
    )
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert isinstance(batch["images"], List)
    assert isinstance(batch["paths"], List)
    assert isinstance(batch["images"][0], Image.Image)
    assert isinstance(batch["paths"][0], str)
