import os 
import sys 
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders import ALL_DATASET_NAMES, load_dataset

table_data = []
for dataset_name in ALL_DATASET_NAMES:
    dataset = load_dataset(dataset_name)
    table_data.append([dataset_name, len(dataset)])

print(tabulate(table_data, headers=['Dataset', 'Length'], tablefmt='grid'))

dataset = load_dataset("gptgeochat")

for idx in range(len(dataset)): 
    item = dataset.__getitem__(idx)
    img = item["image"]
    print(img.size)

    
    break