"""
Give image and prompt and will give generated output
"""
from q_1_steering_vector import create_inputs
from q_2_model import load_model, forward
from PIL import Image
import os

model, processor = load_model()
prompt = "Draft a short factual lede for a news report based on this scene."

folder = ["abalos", "artemisii", "dana", "gas", "olympics_gu", "olympics_liu", "sanchez", "strike", "the_weeknd", "trump", "valldhebron", "wildfire", "zelenski"]
for name in folder: 
    path = os.path.join("q_z_pgd_m_2_7", f"{name}_adversarial.png")
    image = Image.open(path).convert("RGB")

    model_inputs = create_inputs(model, processor, image, prompt)
    decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])

    print(f"❇️ Generated text: {decoded}")