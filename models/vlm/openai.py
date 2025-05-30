"""OpenAI Vision Language Model implementations for image-to-text tasks."""

import openai
from PIL import Image

from models.vlm.base import VLMBaseModel
from utils import pil_to_base64, read_yaml


class OpenAIModel(VLMBaseModel):
    """Base class for OpenAI vision-language models.

    Args:
        private_key (str): OpenAI API key for authentication
        model_name (str): Name of the specific OpenAI model to use
    """

    def __init__(self, private_key, model_name):
        super().__init__(private_key, model_name)
        self.client = openai.OpenAI(api_key=private_key)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        """Process an image with the specified system prompt.

        Args:
            x (Image.Image): Input image to analyze
            system_prompt (str): Instruction prompt for the model

        Returns:
            str: Model's text response
        """
        img_b64 = pil_to_base64(x)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


class GPT4Turbo(OpenAIModel):
    """GPT-4 Turbo vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4-turbo")


class GPT40Mini(OpenAIModel):
    """GPT-4-0 Mini vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4o-mini")


class GPT40(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4o")

class GPT41(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4.1")


class GPT41mini(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4.1-mini")


class GPT41nano(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gpt-4.1-nano")


class GPTO3(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "o3")

class GPTO4mini(OpenAIModel):
    """GPT-4-0 vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "o4-mini")