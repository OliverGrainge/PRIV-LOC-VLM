from typing import Optional
from transformers import pipeline
from PIL import Image
import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import tempfile
from utils import read_yaml

from models.vlm.base import VLMBaseModel


class Qwen25VL7BInstruct(VLMBaseModel):
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
        super().__init__(private_key, "qwen-2.5-vl-7b-instruct")
        config = read_yaml("config.yaml")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", local_files_only=True)



    def forward(self, x: Image.Image, system_prompt: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            x.save(tmp.name)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tmp.name,  # Use file path instead of raw image
                        },
                        {"type": "text", "text": system_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]


class Qwen25VL32BInstruct(VLMBaseModel):
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
        super().__init__(private_key, "qwen-2.5-vl-32b-instruct")
        config = read_yaml("config.yaml")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", local_files_only=True)



    def forward(self, x: Image.Image, system_prompt: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            x.save(tmp.name)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tmp.name,  # Use file path instead of raw image
                        },
                        {"type": "text", "text": system_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]





class Qwen25VL72BInstruct(VLMBaseModel):
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
        super().__init__(private_key, "qwen-2.5-vl-72b-instruct")
        config = read_yaml("config.yaml")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", local_files_only=True)



    def forward(self, x: Image.Image, system_prompt: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            x.save(tmp.name)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tmp.name,  # Use file path instead of raw image
                        },
                        {"type": "text", "text": system_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]