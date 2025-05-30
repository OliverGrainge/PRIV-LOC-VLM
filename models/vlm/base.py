"""Base class for Vision Language Models (VLM) implementations.

This module provides an abstract base class for implementing various vision-language
models that can process images and respond with text based on a system prompt.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from PIL import Image


class VLMBaseModel(nn.Module, ABC):
    """Abstract base class for Vision Language Models.

    This class serves as a template for implementing different VLM models,
    providing a common interface for image-to-text processing.

    Args:
        private_key (str): Authentication key for the model service
        model_name (str): Name identifier for the specific model
    """

    def __init__(self, private_key: str, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.private_key = private_key

    @abstractmethod
    def forward(self, x: Image.Image, system_prompt: str) -> str:
        """Process an image with a system prompt and return a text response.

        Args:
            x (Image.Image): Input image to process
            system_prompt (str): System prompt to guide the model's response

        Returns:
            str: The model's text response
        """
        pass
