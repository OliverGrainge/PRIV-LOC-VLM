import io
from typing import Optional
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch 
from google import genai
from utils import read_yaml

from models.vlm.base import VLMBaseModel


class Gemini(VLMBaseModel):
    """A wrapper class for Google's Gemini vision-language models.

    This class provides an interface to interact with various versions of Gemini
    models that can process both images and text inputs.
    """

    def __init__(self, private_key, model_name):
        """Initialize the Gemini model.

        Args:
            private_key (str): Google API key for authentication
            model_name (str): Name of the specific Gemini model to use
        """
        super().__init__(private_key, model_name)
        self.client = genai.Client(api_key=private_key)
        self.img_bytes_io = io.BytesIO()

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        """Process an image with the specified system prompt.

        Args:
            x (Image.Image): Input image to be processed
            system_prompt (str): Text prompt to guide the model's response

        Returns:
            str: The model's text response
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[x, system_prompt],
        )
        return response.text


class Gemini15Flash8B(Gemini):
    """Gemini 1.5 Flash 8B model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gemini-1.5-flash-8b")


class Gemini15Flash(Gemini):
    """Gemini 1.5 Flash model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gemini-1.5-flash")


class Gemini15Pro(Gemini):
    """Gemini 1.5 Pro model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gemini-1.5-pro")


class Gemini2Flash(Gemini):
    """Gemini 2.0 Flash model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gemini-2.0-flash")


class Gemini2FlashLite(Gemini):
    """Gemini 2.0 Flash Lite model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "gemini-2.0-flash-lite")




# ================================ PaliGemma ================================


class PaliGemma3B224(VLMBaseModel):
    def __init__(self, private_key):
        super().__init__(private_key, "paligemma-3b-mix-224")

        config = read_yaml("config.yaml")

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224", 
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            "google/paligemma-3b-mix-224",
            local_files_only=True,
        )

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        model_inputs = self.model_inputs = self.processor(text=system_prompt, images=x, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]
        generation = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded




