import pytest
from PIL import Image

from models import ALL_MODEL_NAMES
from models.helper import get_vlm_model

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
    return response_raw


@pytest.mark.parametrize(
    "model_name, system_prompt, output_filter_fn",
    [
        (model_name, system_prompt, example_response_filter)
        for model_name in ALL_MODEL_NAMES
    ],
)
def test_model_output_format(model_name, system_prompt, output_filter_fn):
    model = get_vlm_model(
        model_name=model_name,
        system_prompt=system_prompt,
        output_filter_fn=output_filter_fn,
    )
    img = Image.open("models/assets/example_img.jpg")
    response = model(img, system_prompt)
    assert isinstance(response, str)