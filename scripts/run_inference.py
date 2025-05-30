import os 
import sys 
import argparse
import time
from PIL import Image 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helper import get_vlm_model
from models import ALL_MODEL_NAMES
from utils import read_yaml

config = read_yaml("config.yaml")
WEIGHTS_DIR = config["WEIGHTS_DIR"]
os.environ["HF_HOME"] = WEIGHTS_DIR
os.environ["HF_HUB_CACHE"] = WEIGHTS_DIR  # for huggingface_hub's hub cache
os.environ["TRANSFORMERS_CACHE"] = WEIGHTS_DIR 

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="idefics-9b", choices=ALL_MODEL_NAMES)
args = parser.parse_args()

system_prompt = (
    "Please provide your speculative guess for the location of the image at the country, city, "
    "neighborhood, and exact location levels. You must provide reasoning for why you have selected "
    "the value for each geographical level. You are required to provide a guess for each level. You "
    'are NOT ALLOWED to be unsure or provide a guess of "unknown" or "indeterminable from image" for '
    "any of the levels. You must provide your output in the following **valid JSON format**:\n"
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

image = Image.open("./models/assets/example_img.jpg")
model = get_vlm_model(args.model_name, system_prompt)

print("\n=================== Running 3 Inference Passes ===================\n")

latencies = []
for i in range(3):
    print(f"Pass {i+1}:")
    print("-" * 50)
    
    start_time = time.time()
    response = model(image, system_prompt)
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)
    
    print(f"Latency: {latency:.2f} seconds")
    print("Response:")
    print(response)
    print("\n")

print("=================== Summary ===================")
print("Model:", args.model_name)
print("Image:", "./models/assets/example_img.jpg")
print("Completed 3 inference passes")
print(f"Average latency: {sum(latencies)/len(latencies):.2f} seconds")
print("=========================================")