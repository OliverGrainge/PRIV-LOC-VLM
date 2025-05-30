import io
import os

import msgpack
from PIL import Image

from dataloaders.base_dataloader import BaseDataset


class MP16(BaseDataset):
    """Custom dataset to load images from msgpack files"""

    def __init__(self, data_root, image_size=None):
        super().__init__(data_root)
        shard_filenames = sorted(
            os.listdir(data_root), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        shard_filenames = [os.path.join(data_root, f) for f in shard_filenames]
        self.shard_filenames = shard_filenames
        self.image_records = self._load_image_records()

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        record = self.image_records[idx]
        try:
            img_id = record["id"]
            lat, lon = record["latitude"], record["longitude"]
            img = Image.open(io.BytesIO(record["image"])).convert("RGB")

        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            return None

        return {"image": img, "path": img_id}

    def _load_image_records(self):
        """Reads image records from all msgpack files"""
        image_records = []
        for shard_fname in self.shard_filenames:
            with open(shard_fname, "rb") as infile:
                for record in msgpack.Unpacker(infile, raw=False):
                    image_records.append(record)
        return image_records
