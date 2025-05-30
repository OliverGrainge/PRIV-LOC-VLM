import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import tqdm
from torch.utils.data import DataLoader

from dataloaders.helper import collate_fn, load_dataset
from dataloaders.mp16 import MP16
from dataloaders.standarddataset import StandardDataset
from models.helper import get_vlm_model



# dataset_names = ["gptgeochat", "im2gps", "im2gps3k", "yfcc4k"]
dataset_names = ["im2gps3k"]
system_prompt = (
    "Please provide your speculative guess for the location of the image at the country, city, "
    "neighborhood, and exact location levels. You must provide reasoning for why you have selected "
    "the value for each geographical level. You are required to provide a guess for each level. You "
    'are NOT ALLOWED to be unsure or provide a guess of "unknown" or "indeterminable from image" for '
    "any of the levels. Please provide your output in the following JSON format:\n"
    "{\n"
    '    "rationale": "Country: I chose United States as the country because ... City: I chose New York City as the city '
    "because ... Neighborhood: I chose Manhattan as the neighborhood because ... Exact: I chose Empire State "
    'Building as the exact location because ...",\n'
    '    "country": "United States",\n'
    '    "city": "New York City",\n'
    '    "neighborhood": "Manhattan",\n'
    '    "exact_location_name": "Empire State Building",\n'
    '    "latitude": "40.748817",\n'
    '    "longitude": "-73.985428"\n'
    "}\n"
)


def example_response_filter(response):
    return response.strip()


model = get_vlm_model(
    model_name="gpt-4o-mini",
    system_prompt=system_prompt,
    output_filter_fn=example_response_filter,
)

results = []
for dataset_name in dataset_names:
    dataset = load_dataset(dataset_name)
    database_dataloader = DataLoader(
        dataset=dataset, batch_size=4, collate_fn=collate_fn
    )
    for batch in database_dataloader:
        for image, img_pth in zip(batch["images"], batch["paths"]):
            response = model(image, system_prompt)
            results.append([img_pth, response])
        break


# Print results
print("\nModel Responses:")
print("-" * 80)
for img_pth, response in results:
    print(f"\n ========= {img_pth}: ========= ")
    print(f"  {response}")
print("-" * 80)

# Print results
print("\Datasets:")
print("-" * 80)
for dataset_name in dataset_names:
    dataset = load_dataset(dataset_name)
    print(f"\n ========= Dataset: {dataset_name}: Size: {len(dataset)} =========")
print("-" * 80)
