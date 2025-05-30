import os
import pandas as pd
from PIL import Image

from dataloaders.base_dataloader import BaseDataset
from utils import read_img_pths
import random

class OSV5Dataset(BaseDataset):
    def __init__(self, dataset_name="osv5", max_images=500, seed=42):
        super().__init__(dataset_name)
        self.images_dir = os.path.join(self.data_root, "00/")
        self.csv_path = self.metadata_pth  # Use the path from BaseDataset instead of constructing it manually
        self.max_images = max_images
        self.seed = seed
        # Load metadata from test.csv
        self.metadata = pd.read_csv(self.csv_path)

        # Load image records (mapping from image ID to lat/lon)
        self.image_records = self._load_image_records()

        # Load image paths from the directory, filtering by those in test.csv
        self.img_pths = self._get_image_paths()

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

    def _get_image_paths(self):
        """Get filtered and optionally limited image paths"""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        allowed_ids = set(
            os.path.splitext(str(x))[0]
            for x in self.metadata[
                next((col for col in ["id", "IMG_ID", "name", "image_name"] if col in self.metadata.columns))
            ]
        )

        all_paths = []
        for fname in os.listdir(self.images_dir):
            if any(fname.lower().endswith(ext) for ext in img_extensions):
                img_id = os.path.splitext(fname)[0]
                if img_id in allowed_ids:
                    all_paths.append(os.path.join(self.images_dir, fname))

        all_paths = sorted(all_paths)  # ensure deterministic order
        if self.max_images is not None:
            rng = random.Random(self.seed)
            all_paths = rng.sample(all_paths, min(self.max_images, len(all_paths)))

        if not all_paths:
            raise FileNotFoundError(f"No matching images found in {self.images_dir}")

        return all_paths


    def _load_image_records(self):
        """Load image records mapping image ID to (lat, lon) coordinates"""
        # For OSV5, the mapping is:
        # IMG_ID -> "id" 
        # LAT -> "latitude"
        # LON -> "longitude"
        column_mapping = {
            "IMG_ID": ["id", "IMG_ID", "name", "image_name"],
            "LAT": ["latitude", "LAT", "Latitude", "lat"],
            "LON": ["longitude", "LON", "Longitude", "lon"],
        }
        
        standardized_columns = {
            key: next((col for col in names if col in self.metadata.columns), None)
            for key, names in column_mapping.items()
        }
        
        if None in standardized_columns.values():
            missing_cols = [key for key, val in standardized_columns.items() if val is None]
            raise ValueError(
                f"CSV file does not contain the necessary columns. "
                f"Missing: {missing_cols}. Found: {self.metadata.columns.tolist()}"
            )
        
        # Create mapping from image ID to (lat, lon) tuple
        image_records = (
            self.metadata.astype({standardized_columns["IMG_ID"]: str})
            .assign(**{standardized_columns["IMG_ID"]: lambda x: x[standardized_columns["IMG_ID"]] + '.jpg'})
            .set_index(standardized_columns["IMG_ID"])[
                [standardized_columns["LAT"], standardized_columns["LON"]]
            ]
            .apply(tuple, axis=1)
            .to_dict()
        )
        return image_records

    def get_coordinates(self, img_path):
        """Get coordinates for a given image path"""
        # Extract image ID from path (filename without extension)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return self.image_records.get(img_id, None)

    def get_metadata_for_image(self, img_path):
        """Get full metadata row for a given image path"""
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        standardized_columns = {
            "IMG_ID": next((col for col in ["id", "IMG_ID", "name", "image_name"] 
                           if col in self.metadata.columns), None)
        }
        
        if standardized_columns["IMG_ID"]:
            row = self.metadata[self.metadata[standardized_columns["IMG_ID"]] == img_id]
            return row.iloc[0].to_dict() if not row.empty else None
        return None