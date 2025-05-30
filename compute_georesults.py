import os
import re
import pandas as pd
import shutil
from typing import Optional, Tuple
from dataloaders.helper import load_dataset


# ── CONFIGURE THESE CONSTANTS ────────────────────────────────────────────────────
DATASET        = "im2gps"                       # <-- name of the dataset to process
MODELS         = ["gpt-4.1", "gpt-4o", "gemini-1.5-pro", "gemini-2.0-flash", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "qwen-2.5-vl-32b-instruct", "qwen-2.5-vl-72b-instruct"]  # <-- list of model names
RESPONSES_DIR  = "results/responses"             # <-- where {dataset}_{model}_results.csv live
OUTPUT_PATH    = "results/tmp.csv"   # <-- where to write the final CSV
IMAGES_OUTPUT_DIR = "results/images"  # <-- where to save the images
LIMIT         = 100                              # <-- limit number of images to process
# ────────────────────────────────────────────────────────────────────────────────

def extract_coordinates(response: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Pull the first two floats from `response` as (lat, lon).
    Returns (None, None) if missing or out of range.
    """
    s = str(response)
    nums = re.findall(r"-?\d{1,3}\.\d+", s)
    if len(nums) < 2:
        return None, None
    lat, lon = float(nums[0]), float(nums[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    return lat, lon

def load_true_coords(dataset: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by filename with columns ['true_lat','true_lon'].
    Limited to LIMIT number of records if specified.
    """
    ds = load_dataset(dataset)
    records = ds._load_image_records()
    data = {
        fname: {"true_lat": rec[0], "true_lon": rec[1]}
        for i, (fname, rec) in enumerate(records.items())
        if LIMIT is None or i < LIMIT
    }
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "id"
    return df

def load_model_preds(dataset: str, model: str) -> pd.DataFrame:
    """
    Reads RESPONSES_DIR/{dataset}_{model}_results.csv, extracts lat/lon,
    and returns DataFrame indexed by filename with columns
    ['pred_lat_{model}', 'pred_lon_{model}'].
    Limited to the same images as in true_coords.
    """
    path = os.path.join(RESPONSES_DIR, f"{dataset}_{model}_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file: {path}")
    df = pd.read_csv(path, index_col="filename", usecols=["filename","response"])
    
    # Get the true coords to filter by the same images
    true_coords = load_true_coords(dataset)
    df = df[df.index.isin(true_coords.index)]
    
    coords = df["response"].apply(extract_coordinates)
    return pd.DataFrame({
        f"pred_lat_{model}": coords.apply(lambda x: x[0]),
        f"pred_lon_{model}": coords.apply(lambda x: x[1]),
    }, index=df.index).rename_axis("id")

def copy_dataset_images(dataset: str, image_ids: list) -> None:
    """
    Copy images from the dataset to the output directory.
    Args:
        dataset: Name of the dataset
        image_ids: List of image IDs to copy
    """
    ds = load_dataset(dataset)
    os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
    
    for img_path in ds.img_pths:
        # Get the base filename without path
        img_id = os.path.basename(img_path)
        # Remove .jpg extension to match the CSV
        img_id_no_ext = img_id.replace('.jpg', '')
        
        if img_id_no_ext in image_ids:
            dest_path = os.path.join(IMAGES_OUTPUT_DIR, img_id)
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_id} to {dest_path}")

def main():
    # 1) load ground-truth
    true_df = load_true_coords(DATASET)

    # 2) for each model, load preds and join
    combined = true_df.copy()
    for m in MODELS:
        try:
            preds = load_model_preds(DATASET, m)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            # create empty cols
            combined[f"pred_lat_{m}"] = None
            combined[f"pred_lon_{m}"] = None
            continue
        combined = combined.join(preds, how="left")

    # 3) write out
    combined = combined.reset_index()  # make 'id' a column
    # Remove .jpg suffix from id column
    combined['id'] = combined['id'].str.replace('.jpg', '')
    
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {combined.shape[0]} rows × {combined.shape[1]} cols to {OUTPUT_PATH}")

    # 4) Copy corresponding images
    print(f"Copying images to {IMAGES_OUTPUT_DIR}...")
    copy_dataset_images(DATASET, combined['id'].tolist())

if __name__ == "__main__":
    main()
