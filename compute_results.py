import argparse
import os
import re
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate
from math import radians, sin, cos, sqrt, atan2

from dataloaders.helper import load_dataset
from dataloaders import ALL_DATASET_NAMES
from models import ALL_MODEL_NAMES
from pathlib import Path

# Constants
EARTH_RADIUS_KM = 6371.0088
RECALL_DISTANCES = [1, 5, 10, 20, 100, 200, 750]
CHECKPOINT_PATH = "results/metrics/checkpoint.json"

@dataclass
class Coordinates:
    lat: Optional[float]
    lon: Optional[float]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM model on geolocation datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["im2gps"],
        choices=ALL_DATASET_NAMES,
        help="Names of the datasets to evaluate",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+", 
        default=ALL_MODEL_NAMES,
        choices=ALL_MODEL_NAMES,
        help="Names of the VLM models to use",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the metrics to a csv file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last checkpoint if available",
    )
    return parser.parse_args()

def extract_coordinates(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract latitude and longitude from a response string."""
    # Convert response to string if it's not already a string
    response_str = str(response)
    
    numbers = re.findall(r"-?\d{1,3}\.\d+", response_str)
    if len(numbers) < 2:
        return None, None
        
    lat, lon = float(numbers[0]), float(numbers[1])
    
    # Validate coordinate ranges
    valid_lat = -90 <= lat <= 90
    valid_lon = -180 <= lon <= 180

    
    return (
        numbers[0] if valid_lat else None,
        numbers[1] if valid_lon else None
    )

def get_results(model_name: str, dataset_name: str) -> pd.DataFrame:
    """Load and process results from a CSV file."""
    rel_path = f"results/responses/{dataset_name}_{model_name}_results.csv"
    pth = os.path.join(os.path.dirname(__file__), rel_path)
    print(f"Loading results from: {pth}")
    
    df = pd.read_csv(pth).set_index('filename')
    lat, lon = zip(*df['response'].apply(extract_coordinates))
    df['lat'] = lat
    df['lon'] = lon

    return df

def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """Calculate the great circle distance between two points on Earth."""
    if any(v is None for v in [coord1.lat, coord1.lon, coord2.lat, coord2.lon]):
        return float('inf')
    
    lat1, lon1 = map(float, [coord1.lat, coord1.lon])
    lat2, lon2 = map(float, [coord2.lat, coord2.lon])
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return EARTH_RADIUS_KM * c

def compute_distances(df: pd.DataFrame, records: Dict) -> List[float]:
    """Compute distances between predicted and ground truth coordinates."""
    distances = []
    
    for filename, row in df.iterrows():
        try:
            pred_coords = Coordinates(
                lat=float(row['lat']) if row['lat'] is not None else None,
                lon=float(row['lon']) if row['lon'] is not None else None
            )
            #print("=========" * 100, filename, list(records.keys())[0], type(filename), type(list(records.keys())[0]))
            #raise Exception("Stop here")
            true_coords = Coordinates(
                lat=records[filename][0],
                lon=records[filename][1]
            )
            dist = haversine_distance(pred_coords, true_coords)
            distances.append(dist)
        except:
            print("======== ERROR with Filename: ", filename, "Becuase of pred: ", pred_coords, "and true: ", true_coords)
            
    return distances

def calculate_recall(distances: List[float], threshold_km: float) -> float:
    """Calculate recall at K kilometers."""
    if not distances:
        return 0.0
    return (sum(1 for d in distances if d <= threshold_km) / len(distances)) * 100

def load_checkpoint() -> Dict:
    """Load checkpoint data if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Error reading checkpoint file. Starting fresh.")
    return {"completed_items": []}

def save_checkpoint(checkpoint_data: Dict) -> None:
    """Save checkpoint data."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint_data, f)

def main() -> None:
    args = parse_args()
    
    # Load checkpoint if resume flag is set
    checkpoint_data = load_checkpoint() if args.resume else {"completed_items": []}
    completed_items = set(checkpoint_data["completed_items"])
    
    # Create a directory for metrics if it doesn't exist
    os.makedirs("results/metrics", exist_ok=True)
    
    try:
        for dataset in args.datasets:
            ds = load_dataset(dataset)
            records = ds._load_image_records()
            
            print(f"\nResults for {dataset}:")
            
            headers = ["Model"] + [f"R@{k}km" for k in RECALL_DISTANCES]
            table_data = []
            
            for model in args.models:
                # Skip if this combination was already processed
                checkpoint_key = f"{dataset}_{model}"
                if checkpoint_key in completed_items:
                    print(f"Skipping already processed: {model} on {dataset}")
                    
                    # Try to load the previously calculated results
                    try:
                        metrics_path = Path("results/metrics") / f"{dataset}_metrics.csv"
                        if metrics_path.exists():
                            metrics_df = pd.read_csv(metrics_path)
                            model_row = metrics_df[metrics_df['Model'] == model]
                            if not model_row.empty:
                                row_data = model_row.values.tolist()[0]
                                table_data.append(row_data)
                            continue
                    except Exception as e:
                        print(f"Error loading previous results: {e}. Reprocessing.")
                
                try:
                    df = get_results(model, dataset)
                except FileNotFoundError:
                    print(f"No results found for {model} on {dataset}")
                    continue
                distances = compute_distances(df, records)
                
                recalls = [
                    calculate_recall(distances, k) for k in RECALL_DISTANCES
                ]
                
                row_data = [model] + [f"{recall:.3f}" for recall in recalls]
                table_data.append(row_data)
                
                # Mark as completed and save checkpoint
                completed_items.add(checkpoint_key)
                checkpoint_data["completed_items"] = list(completed_items)
                save_checkpoint(checkpoint_data)
                
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print("\n")
            
            if args.save and table_data:
                results_df = pd.DataFrame(table_data, columns=headers)
                save_path = Path("results/metrics") / f"{dataset}_metrics.csv"
                save_path.parent.mkdir(exist_ok=True)
                results_df.to_csv(save_path, index=False)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        # Save checkpoint before exiting, even on error
        checkpoint_data["completed_items"] = list(completed_items)
        save_checkpoint(checkpoint_data)
        raise
    
    # Clear checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH) and len(completed_items) == len(args.datasets) * len(args.models):
        os.remove(CHECKPOINT_PATH)
        print("Processing completed successfully. Checkpoint cleared.")

if __name__ == "__main__":
    main()