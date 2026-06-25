"""
FILE THAT LOADS THE QWEN2-VL MODEL AND DOES A FORWARD PASS.

"""

from transformers import (
    AutoProcessor, 
    Qwen2VLForConditionalGeneration, #Qwen2-VL-7B-Instruct
)   

import torch

def load_model (device="cuda:0"):
    id = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(id, torch_dtype=torch.float16, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(
        id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28   # cap at 512 patches
    )
    # It handles tokenization and image preprocessing with pixel count between 256 and 512 patches (28x28 pixels per patch).

    return model, processor

def forward(model, processor, model_inputs, max_new_tokens=150, do_sample=False, temperature=None, top_p=None, top_k=None, input_len=None, repetition_penalty=1.3):
    generation = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)
    # (1, input_len + generated_len)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded