import os

import pytest
from PIL import Image

from dataloaders import ALL_DATASET_NAMES
from dataloaders.helper import collate_fn, load_dataset


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_records_instantiation(dataset_name):
    dataset = load_dataset(dataset_name)
    records = dataset._load_image_records()
    assert records is not None


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_records_types(dataset_name):
    dataset = load_dataset(dataset_name)
    records = dataset._load_image_records()
    assert isinstance(records, dict)
    assert isinstance(list(records.keys())[0], str)
    assert isinstance(records[list(records.keys())[0]], tuple)
    assert len(records[list(records.keys())[0]]) == 2
    assert isinstance(records[list(records.keys())[0]][0], float)
    assert isinstance(records[list(records.keys())[0]][1], float)


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_records_lengths(dataset_name):
    dataset = load_dataset(dataset_name)
    records = dataset._load_image_records()
    record_img_names = set(os.path.basename(name) for name in records.keys())
    dataset_img_names = set(os.path.basename(pth) for pth in dataset.img_pths)

    missing_images = dataset_img_names - record_img_names
    num_missing = len(missing_images)

    assert (
        num_missing == 0
    ), f"{num_missing} dataset ({dataset_name}) images are missing from records. First few missing: {list(missing_images)[:3]}"
