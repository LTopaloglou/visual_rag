from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List

class VisualQA:

    def __init__(self, device: str, max_tokens: int = 128):
        self.device = device
        self.max_tokens = max_tokens
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype="torch.bfloat16", 
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def predict(self, image_paths: List[str], queries: List[str]) -> str:
        image_messages = [{"type": "image", "image": path} for path in image_paths]
        query_messages = [{"type:": "text", "text": query} for query in queries]
        message = {
            "role": "user",
            "content": image_messages + query_messages,
        }
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([message])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text