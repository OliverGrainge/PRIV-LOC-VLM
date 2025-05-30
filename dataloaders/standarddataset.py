import os

import pandas as pd
from PIL import Image

from dataloaders.base_dataloader import BaseDataset
from utils import read_img_pths


class StandardDataset(BaseDataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.img_pths = read_img_pths(self.data_root)
        self.metadata = pd.read_csv(self.metadata_pth)
        self.image_records = self._load_image_records()

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        img_pth = self.img_pths[idx]
        try:
            img = Image.open(img_pth)
            img = img.resize((256, 256))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_pth}: {e}")
        return {"image": img, "path": img_pth}

    def _load_image_records(self):
        column_mapping = {
            "IMG_ID": ["IMG_ID", "name", "image_name"],
            "LAT": ["LAT", "Latitude", "lat"],
            "LON": ["LON", "Longitude", "lon"],
        }
        standardized_columns = {
            key: next((col for col in names if col in self.metadata.columns), None)
            for key, names in column_mapping.items()
        }
        if None in standardized_columns.values():
            raise ValueError(
                f"CSV file does not contain the necessary columns. Found: {self.metadata.columns.tolist()}"
            )
        image_records = (
            self.metadata.set_index(standardized_columns["IMG_ID"])[
                [standardized_columns["LAT"], standardized_columns["LON"]]
            ]
            .apply(tuple, axis=1)
            .to_dict()
        )
        return image_records
