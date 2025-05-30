import argparse
import csv
import os
import pandas as pd
import torch
import tqdm

from torch.utils.data import DataLoader
from dataloaders.helper import collate_fn, load_dataset
from models import ALL_MODEL_NAMES
from models.helper import get_vlm_model
from utils import read_yaml
# ----------------------------
# 1. Argument parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM model on geolocation datasets (with resume support)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="im2gps",
        choices=[
            "im2gps3k",
            "im2gps",
            "yfcc4k",
            "gptgeochat",
            "MP16",
            "osv5",
        ],
        help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-o4-mini",
        choices=ALL_MODEL_NAMES,
        help="Name of the VLM model to use",
    )
    return parser.parse_args()

# ----------------------------
# 2. Your system prompt & filter
# ----------------------------
system_prompt = (
    "Please provide your speculative guess for the location of the image at the country, city, "
    "neighborhood, and exact location levels. You must provide reasoning for why you have selected "
    "the value for each geographical level. You are required to provide a guess for each level. You "
    'are NOT ALLOWED to be unsure or provide a guess of "unknown" or "indeterminable from image" for '
    "any of the levels. You must provide your output in the following **valid JSON format**:\n"
    "{\n"
    '    "rationale": "Country: I chose United States because ...",\n'
    '    "country": "United States",\n'
    '    "city": "New York City",\n'
    '    "neighborhood": "Manhattan",\n'
    '    "exact_location_name": "Empire State Building",\n'
    '    "latitude": "40.748817",\n'
    '    "longitude": "-73.985428"\n'
    "}\n"
)

def response_filter(response):
    # (Optionally parse or clean up the model's JSON)
    return response

# ----------------------------
# 3. Main with resume logic
# ----------------------------
def main():
    args = parse_args()

    out_dir = "results/responses"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.dataset}_{args.model}_results.csv")

    # 3a) Load any existing results to skip them
    processed = set()
    if os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path)
        processed = set(df_prev["filename"].astype(str).tolist())

    # 3b) Prepare CSV for appending; write header if new
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["filename", "system_prompt", "response"])
    if not processed:
        writer.writeheader()

    # 3c) Initialize model
    model = get_vlm_model(
        model_name=args.model,
        system_prompt=system_prompt,
        output_filter_fn=response_filter,
    )

    dataset = load_dataset(args.dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)

    try:
        for batch in tqdm.tqdm(dataloader, desc=f"Processing {args.dataset}"):
            for image, img_pth in zip(batch["images"], batch["paths"]):
                base_fn = os.path.basename(img_pth)
                if base_fn in processed:
                    # skip already-done examples
                    continue

                # run model
                with torch.no_grad():
                    resp = model(image, system_prompt)

                # write out immediately
                writer.writerow({
                    "filename": base_fn,
                    "system_prompt": system_prompt,
                    "response": resp,
                })
                csv_file.flush()
                processed.add(base_fn)

    except Exception as e:
        # on crash, close file and re-raise so you see the error
        csv_file.close()
        raise

    csv_file.close()
    print(f"✅ Completed. Results saved to {csv_path}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = config["MODEL_KEYS"]["HUGGINGFACE"]
    os.environ["HF_HOME"] = config["WEIGHTS_DIR"]
    main()
