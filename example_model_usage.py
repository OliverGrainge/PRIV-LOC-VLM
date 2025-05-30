import json
import sys
from PIL import Image
import os
import sys

from models import ALL_MODEL_NAMES  # List of available model names
from models.helper import \
    get_vlm_model  # Function to initialize vision-language models

# Load the example image
img = Image.open("models/assets/example_img.jpg")

# Define the system prompt that instructs the models how to process the image
# This prompt specifically asks for location estimation in lat/long format
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


def example_response_filter(response_raw):
    response_raw = response_raw.strip()
    try:
        if isinstance(response_raw, str):
            clean_str = response_raw.strip()
            # Handle markdown code blocks
            if clean_str.startswith("```"):
                lines = clean_str.splitlines()
                # Remove opening and closing code block markers
                lines = [line for line in lines if not line.startswith("```")]
                clean_str = "\n".join(lines).strip()

            # Find the JSON object within the text
            start_idx = clean_str.find("{")
            if start_idx != -1:
                end_idx = clean_str.rfind("}")
                if end_idx != -1:
                    clean_str = clean_str[start_idx : end_idx + 1]
                else:
                    raise json.JSONDecodeError("No closing brace found", clean_str, 0)
            else:
                raise json.JSONDecodeError("No JSON object found", clean_str, 0)

            response = json.loads(clean_str)
            return response
        else:
            return response_raw
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}", "raw_response": response_raw}


results = []

# Iterate through each available model
for model_name in ["phi-4-multimodal-instruct",]:
    # Initialize the current vision-language model with specified parameters
    model = get_vlm_model(
        model_name=model_name,
        system_prompt=system_prompt,
        output_filter_fn=example_response_filter,
    )

    # Process the image with the current model and get its response
    response = model(img, system_prompt)

    # Store the model name and its response
    results.append([model_name, response])

# Print results
print("\nModel Responses:")
print("-" * 80)
for model_name, response in results:
    print(f"\n ========= {model_name}: ========= ")
    print(f"  {response}")
print("-" * 80)
