import json
import os
from glob import glob

import pandas as pd
from PIL import Image

from dataloaders.base_dataloader import BaseDataset
from utils import read_img_pths


class GPTGeoChat(BaseDataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.img_pths = read_img_pths(self.data_root)
        with open(self.metadata_pth, "r") as f:
            self.metadata = json.load(f)
        self.image_records = self._load_image_records()

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        img_pth = self.img_pths[idx]
        try:
            img = Image.open(img_pth)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_pth}: {e}")
        return {"image": img, "path": img_pth}

    def _load_image_records(self):
        image_records = {
            img_id
            + ".jpg": (
                float(details["latitude"]) if details["latitude"] else None,
                float(details["longitude"]) if details["longitude"] else None,
            )
            for img_id, details in self.metadata.items()
        }

        # Optional: Remove entries where lat/lon are None
        # image_records = {k: v for k, v in image_records.items() if None not in v}
        return image_records
