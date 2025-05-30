import openai
from PIL import Image
import anthropic
import io 
import base64
from models.vlm.base import VLMBaseModel
from utils import pil_to_base64, read_yaml
from groq import Groq
from transformers import MllamaForConditionalGeneration, AutoProcessor 
import torch 
import time
from typing import Optional
import logging

from groq import Groq
import base64
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import pipeline
from transformers import AutoModelForImageTextToText




class Llama32_11B(VLMBaseModel):
    """Llama 3.2 11B Vision Instruct model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "Llama-3.2-11B-Vision-Instruct")
        config = read_yaml("config.yaml")
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": system_prompt}
                ]
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        model_inputs = self.processor(
            x,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        # Extract just the input_ids for generation
        output = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=500
        )
        full_response = self.processor.decode(output[0])
        
        # Extract only the assistant's response
        assistant_start = full_response.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            response = full_response[assistant_start + len("<|start_header_id|>assistant<|end_header_id|>"):].strip()
            # Remove any trailing EOT marker if present
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            return response
        return full_response  # Fallback to full response if we can't find the assistant marker


class Llama32_90B(VLMBaseModel):
    """Llama 3.2 90B Vision Instruct model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "Llama-3.2-90B-Vision-Instruct")
        config = read_yaml("config.yaml")
        self.pipe = pipeline("image-text-to-text", model="meta-llama/Llama-3.2-90B-Vision-Instruct", device_map="auto", local_files_only=True)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": x},
                    {"type": "text", "text": system_prompt}
                ]
            },
        ]
        records = self.pipe(text=messages, max_new_tokens=500)
        assistant_replies = [
            msg['content']
            for record in records
            for msg    in record['generated_text']
            if msg.get('role') == 'assistant'
        ]
        return assistant_replies[0]




class LlamaScout17B16E(VLMBaseModel):
    """Llama Scout 17B 16E Vision Instruct model implementation."""

    def __init__(self, private_key):
        super().__init__(private_key, "llama-4-scout-17b-16e-instruct")
        config = read_yaml("config.yaml")
        model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def forward(self, x: Image.Image, system_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": system_prompt}
                ]
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        model_inputs = self.processor(
            x,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        # Extract just the input_ids for generation
        output = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=500
        )
        full_response = self.processor.decode(output[0])
        
        # Extract only the assistant's response
        assistant_start = full_response.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            response = full_response[assistant_start + len("<|start_header_id|>assistant<|end_header_id|>"):].strip()
            # Remove any trailing EOT marker if present
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            return response
        return full_response  # Fallback to full response if we can't find the assistant marker
