import openai
from PIL import Image
import anthropic
import io 
import base64
from models.vlm.base import VLMBaseModel
from utils import pil_to_base64, read_yaml


class CLAUDE(VLMBaseModel):
    """Base class for OpenAI vision-language models.

    Args:
        private_key (str): OpenAI API key for authentication
        model_name (str): Name of the specific OpenAI model to use
    """

    def __init__(self, private_key, model_name):
        super().__init__(private_key, model_name)
        self.client = anthropic.Anthropic(api_key=private_key, max_retries=3, timeout=1200)
        self.img_bytes_io = io.BytesIO()

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        """Process an image with the specified system prompt.

        Args:
            x (Image.Image): Input image to analyze
            system_prompt (str): Instruction prompt for the model

        Returns:
            str: Model's text response
        """
        self.img_bytes_io.seek(0)
        self.img_bytes_io.truncate()
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x.save(self.img_bytes_io, format='JPEG')
        img_str = base64.b64encode(self.img_bytes_io.getvalue()).decode('utf-8')

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_str,
                            },
                        }
                    ],
                }
            ],
        )
        return message.content[0].text


class Sonnet35(CLAUDE):
    """GPT-4 Turbo vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "claude-3-5-sonnet-latest")


class Sonnet37(CLAUDE):
    """GPT-4 Turbo vision model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "claude-3-7-sonnet-latest")