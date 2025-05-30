import argparse
import os
import sys
import pandas as pd
import tqdm

# ensure project root is on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ALL_MODEL_NAMES
from dataloaders.helper import load_dataset


def get_dataset_filenames(dataset_name: str):
    """
    Load the dataset and return the set of all image basenames.
    """
    dataset = load_dataset(dataset_name)
    fnames = set()
    for item in dataset:
        path = item.get("path") or item.get("paths")
        if isinstance(path, (list, tuple)):
            for p in path:
                fnames.add(os.path.basename(p))
        else:
            fnames.add(os.path.basename(path))
    return fnames


def check_results_for_model(dataset_name: str, model_name: str, result_dir: str, all_fnames: set):
    """
    Check a single model's CSV for existence, completeness, and non-empty responses.
    Uses a tqdm bar over each image to show per-image checking progress.
    Returns a dict with keys: status, missing, extra, empty.
    status is one of: 'missing_csv', 'incomplete', 'complete'
    """
    csv_name = f"{dataset_name}_{model_name}_results.csv"
    csv_path = os.path.join(result_dir, csv_name)

    if not os.path.isfile(csv_path):
        return {"status": "missing_csv"}

    df = pd.read_csv(csv_path, usecols=["filename", "response"]).set_index("filename")
    seen = set(df.index.tolist())

    missing = []
    empty = []
    # progress bar for each image
    for fname in tqdm.tqdm(sorted(all_fnames),
                           desc=f"Checking images for {dataset_name}-{model_name}",
                           position=2,
                           leave=False):
        if fname not in seen:
            missing.append(fname)
        else:
            resp = str(df.at[fname, "response"]).strip()
            if not resp:
                empty.append(fname)

    extra = sorted(seen - all_fnames)

    if missing or extra or empty:
        return {
            "status": "incomplete",
            "missing": missing,
            "extra": extra,
            "empty": empty,
        }

    return {"status": "complete"}


def main():
    parser = argparse.ArgumentParser(
        description="Check that each model has a full set of responses for a dataset, with nested progress bars"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=["im2gps"],
        help="Names of datasets to check",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/",
        help="Root directory containing the `responses/` subfolder",
    )
    args = parser.parse_args()

    result_dir = os.path.join(args.results_dir, "responses")
    exit_code = 0

    # Outer progress bar for datasets
    for ds in tqdm.tqdm(args.dataset_names, desc="Datasets", position=0):
        print(f"\nDataset: {ds}")
        all_fnames = get_dataset_filenames(ds)

        missing_models = []
        incomplete_models = []
        complete_models = []

        # Inner progress bar for each model/file
        pbar = tqdm.tqdm(ALL_MODEL_NAMES,
                         desc=f"Models for {ds}",
                         position=1,
                         leave=False)
        for model_name in pbar:
            pbar.set_postfix(model=model_name)
            info = check_results_for_model(ds, model_name, result_dir, all_fnames)
            status = info["status"]
            if status == "missing_csv":
                missing_models.append(model_name)
            elif status == "incomplete":
                incomplete_models.append((model_name, info))
            else:
                complete_models.append(model_name)
        pbar.close()

        # Summary for this dataset
        if complete_models:
            print("\n")
            print(f"  ✓ Complete results for {len(complete_models)} models:")
            print("    ", ", ".join(complete_models))
            print("\n")
        if missing_models:
            print("\n")
            print(f"  ✗ Missing CSV for {len(missing_models)} models:")
            print("    ", ", ".join(missing_models))
            exit_code = 1
            print("\n")
        if incomplete_models:
            print("\n")
            print(f"  ⚠️ Incomplete results for {len(incomplete_models)} models:")
            for model_name, info in incomplete_models:
                print(f"    - {model_name}: {len(info['missing'])} missing, {len(info['extra'])} extra, {len(info['empty'])} empty responses")
            print("\n")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
