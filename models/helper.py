import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
from PIL import Image

from models.vlm import *
from utils import read_yaml
import os 


def default_filter_response(response: str) -> str:
    """Default filter function that returns the response unchanged.

    Args:
        response (str): The model's response to filter

    Returns:
        str: The unmodified response
    """
    return response


class VLMBase(nn.Module):
    """Base class for Vision Language Models implementing a common interface.

    Args:
        vlm_model (nn.Module): The underlying vision language model
        system_prompt (str): Default system prompt to use with the model
        output_filter_fn (Callable[[str], str]): Function to process model outputs
    """

    def __init__(
        self,
        vlm_model: nn.Module,
        system_prompt: str,
        output_filter_fn: Callable[[str], str],
    ) -> None:
        super().__init__()
        self.vlm_model = vlm_model
        self.system_prompt = system_prompt
        self.output_filter_fn = output_filter_fn

    def forward(self, x: Image.Image, system_prompt: Optional[str] = None) -> str:
        """Process an image with the vision language model.

        Args:
            x (Image.Image): Input image to process
            system_prompt (Optional[str], optional): Override default system prompt.
                Defaults to None.

        Returns:
            str: Processed model response
        """
        if system_prompt is not None:
            response = self.vlm_model(x, system_prompt)
        else:
            response = self.vlm_model(x, self.system_prompt)
        return self.output_filter_fn(response)


def get_vlm_model(
    model_name: str,
    system_prompt: str,
    output_filter_fn: Callable[[str], str] = default_filter_response,
) -> VLMBase:
    """Factory function to create a VLM model instance.

    Args:
        model_name (str): Name of the model to instantiate
        system_prompt (str): System prompt to use with the model
        output_filter_fn (Callable[[str], str], optional): Function to process model outputs.
            Defaults to default_filter_response.

    Raises:
        ValueError: If the requested model name is not found in the supported models

    Returns:
        VLMBase: Instantiated vision language model
    """
    config = read_yaml("config.yaml")

    model_mapping: dict[str, Type[nn.Module]] = {
        "gpt-4o": GPT40,
        "gpt-4.1": GPT41,
        "gpt-4.1-mini": GPT41mini,
        "gpt-4.1-nano": GPT41nano,
        "gpt-o3": GPTO3,
        "gpt-o4-mini": GPTO4mini,
        "gemini-1.5-flash-8b": Gemini15Flash8B,
        "gemini-1.5-flash": Gemini15Flash,
        "gemini-1.5-pro": Gemini15Pro,
        "gemini-2.0-flash": Gemini2Flash,
        "gemini-2.0-flash-lite": Gemini2FlashLite,
        "phi-4-multimodal-instruct": Phi4,
        "claude-3-5-sonnet-latest": Sonnet35,
        "claude-3-7-sonnet-latest": Sonnet37,
        "llama-3.2-90b-vision-instruct": Llama32_90B,
        "llama-3.2-11b-vision-instruct": Llama32_11B,
        "llama-4-scout-17b-16e-instruct": LlamaScout17B16E,
        "idefics-9b-instruct": Idefics9B,
        "idefics-80b-instruct": Idefics80B,
        "paligemma-3b-mix-224": PaliGemma3B224,
        "qwen-2.5-vl-7b-instruct": Qwen25VL7BInstruct,
        "qwen-2.5-vl-32b-instruct": Qwen25VL32BInstruct,
        "qwen-2.5-vl-72b-instruct": Qwen25VL72BInstruct,
    }

    model_class = model_mapping.get(model_name)
    if not model_class:
        raise ValueError(f"Model {model_name} not found")

    if "gpt" in model_name or model_name == "o1":
        api_key = config["MODEL_KEYS"]["OPENAI_KEY"]
    elif "gemini" in model_name:
        api_key = config["MODEL_KEYS"]["GOOGLE_KEY"]
    elif model_name == "phi-4-multimodal-instruct":
        api_key = config["MODEL_KEYS"]["MICROSOFT_KEY"]
    elif "sonnet" in model_name: 
        api_key = config["MODEL_KEYS"]["ANTHROPIC_KEY"]
    elif "llama" in model_name:
        api_key = config["MODEL_KEYS"]["GROQ_KEY"]
    else:
        api_key = None

    return VLMBase(model_class(api_key), system_prompt, output_filter_fn)
