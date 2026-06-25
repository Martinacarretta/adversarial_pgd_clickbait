"""
Give image and prompt and will give generated output
"""
from adversarial_pgd_clickbait.q_1_steering_vector import create_inputs
from adversarial_pgd_clickbait.q_2_model import load_model, forward
from PIL import Image
import os

model, processor = load_model()

name = "olympics_gu" # "olympics_gu", "gas", "wildfire", "strike" # "dana", "trump", "artemisii"
path = os.path.join("q_z_pgd", f"{name}_adversarial.png")
image = Image.open(path).convert("RGB")

prompt = "What's in this image?"

model_inputs = create_inputs(model, processor, image, prompt)
decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])

print(f"Generated text: {decoded}")