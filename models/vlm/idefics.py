from transformers import AutoProcessor, IdeficsForVisionText2Text

from typing import Optional

from google import genai
from PIL import Image
import torch 
from transformers import pipeline
from utils import read_yaml

from models.vlm.base import VLMBaseModel


class Idefics9B(VLMBaseModel):
    def __init__(self, private_key: Optional[str] = None):
        super().__init__(None, "IDEFICS-9B")
        config = read_yaml("config.yaml")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        prompts = [
            [
                "User:", 
                x, 
                system_prompt + " \nAssistant: "    
            ]
        ]
        
        # Process inputs
        inputs = self.processor(
            prompts, 
            add_end_of_utterance_token=False, 
            return_tensors="pt"
        ).to(self.device)
        
        # Set up generation parameters
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], 
            add_special_tokens=False
        ).input_ids
        
        # Generate response
        generated_ids = self.model.generate(
            **inputs, 
            eos_token_id=exit_condition, 
            bad_words_ids=bad_words_ids, 
            max_length=1000
        )
        
        # Decode and extract response
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = generated_text.split("Assistant:")[-1].strip()
        return response




class Idefics80B(VLMBaseModel):
    def __init__(self, private_key: Optional[str] = None):
        super().__init__(None, "IDEFICS-80B")
        config = read_yaml("config.yaml")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        prompts = [
            [
                "User:", 
                x, 
                system_prompt + " \nAssistant: "    
            ]
        ]
        
        # Process inputs
        inputs = self.processor(
            prompts, 
            add_end_of_utterance_token=False, 
            return_tensors="pt"
        ).to(self.device)
        
        # Set up generation parameters
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], 
            add_special_tokens=False
        ).input_ids
        
        # Generate response
        generated_ids = self.model.generate(
            **inputs, 
            eos_token_id=exit_condition, 
            bad_words_ids=bad_words_ids, 
            max_length=1000
        )
        
        # Decode and extract response
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = generated_text.split("Assistant:")[-1].strip()
        return response
        
        
