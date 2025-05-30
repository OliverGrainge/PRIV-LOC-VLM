import io
from typing import Optional

import base64
from PIL import Image
import requests

from models.vlm.base import VLMBaseModel



class Phi4(VLMBaseModel):
    """A wrapper class for Google's Gemini vision-language models.

    This class provides an interface to interact with various versions of Gemini
    models that can process both images and text inputs.
    """

    def __init__(self, private_key):
        """Initialize the Gemini model.

        Args:
            private_key (str): Google API key for authentication
            model_name (str): Name of the specific Gemini model to use
        """
        super().__init__(private_key, "Phi-4-multimodal-instruct")
        self.endpoint = "https://privlockresour3165674654.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
        self.img_bytes_io = io.BytesIO()

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        """Process an image with the specified system prompt.

        Args:
            x (Image.Image): Input image to be processed
            system_prompt (str): Text prompt to guide the model's response

        Returns:
            str: The model's text response
        """
        # Reset the BytesIO object before using it
        self.img_bytes_io = io.BytesIO()
        # Explicitly specify JPEG format when saving
        x.save(self.img_bytes_io, format="JPEG")
        img_str = base64.b64encode(self.img_bytes_io.getvalue()).decode('utf-8')

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and infer its location."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "top_p": 1,
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": self.private_key,
            "x-ms-model-mesh-model-name": self.model_name
        }

        response = requests.post(self.endpoint, headers=headers, json=data)

        if response.status_code == 200:
            output = response.json()
            reply = output["choices"][0]["message"]["content"]
            return reply 
        else:
            print("Request failed:", response.status_code, response.text)

