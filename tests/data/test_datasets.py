import pytest
from PIL import Image

from dataloaders import ALL_DATASET_NAMES
from dataloaders.helper import collate_fn, load_dataset


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_instantiation(dataset_name):
    dataset = load_dataset(dataset_name)
    assert dataset is not None


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_output_format(dataset_name):
    dataset = load_dataset(dataset_name)
    output = dataset.__getitem__(0)
    assert isinstance(output, dict)
    assert "image" in output.keys()
    assert "path" in output.keys()


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_dataset_output_types(dataset_name):
    dataset = load_dataset(dataset_name)
    output = dataset.__getitem__(0)
    assert isinstance(output, dict)
    assert isinstance(output["image"], Image.Image)
    assert isinstance(output["path"], str)
