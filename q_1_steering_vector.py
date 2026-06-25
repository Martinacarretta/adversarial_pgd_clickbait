"""
FILE THAT GETS THE HIDDEN STATES FROM THE MODEL AND COMPUTES THE STEERING VECTOR.
 - Get hidden states: does a single pass without the generation part and gets hidden states of each layer, averages across the sequence dimension to get a single vector per layer.
 - Get steering vector: gets hidden states for multiple polarized prompts, averages them to get a single vector for positive and negative, then subtracts to get the steering vector.
 - Create inputs: creates the model inputs for a given image and user prompt, to be used (since user prompt is plain text)
"""

import torch
from PIL import Image
from adversarial_pgd_clickbait.q_2_model import load_model
import os
import pandas as pd

def create_inputs(model, processor, image, user_prompt): # for main.py
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}]
        }
    ]

    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    model_inputs = {
        k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v
        for k, v in model_inputs.items()
    }
    return model_inputs

def get_hidden_states(image, model, processor, user_prompt):
    messages = [ # build message structure with image and prompt
        {
            "role": "user", 
            "content": 
                [{"type": "image", 
                  "image": image},
                 
                {"type": "text", 
                 "text": user_prompt}]
        }
    ]
    
    # get text with special tokens and add the generation token so model knows
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # tokenize text, process image into tensor and move to GPU
    model_inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    model_inputs = { # convert to float16 if it's 32
        k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v
        for k, v in model_inputs.items()
    }

    # tokens are of dimension 3584
    with torch.inference_mode():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    # outputs a single forward pass without generation through the model.
    # output.hidden_states is a list of hidden states from each layer, including the embedding layer at index 0.
        
    # Stack into (num_layers, seq_len, hidden_size), take mean over seq_len
    # outputs.hidden_states[0] is the embedding layer, [1:] are transformer layers
    layer_means = torch.stack([
        h[0].mean(dim=0)  # [0] because it's a single batch # shape: (seq_len, 3584), mean over seq_len to get (3584,)
        for h in outputs.hidden_states[1:]  # 28 layers
    ])  # shape: (28, 3584) --> for each layer, average the hidden states across the TOKENS to get a single vector per layer.
    # for each of the 28 layers, takes hidden state tensor of shape (1, seq_len, 3584), 
    # TODO: check if we should only take text, not the image tokens, 
    return layer_means # (num_layers, hidden_size)

def get_steering_vector(image, model, processor, polarized_positive, polarized_negative):
    pos_states = [get_hidden_states(image, model, processor, p) for p in polarized_positive] # (num_prompts, num_layers, hidden_size)
    neg_states = [get_hidden_states(image, model, processor, p) for p in polarized_negative] # (num_prompts, num_layers, hidden_size)
        
    vector_pos = torch.stack(pos_states).mean(dim=0)   # (28, 3584)
    vector_neg = torch.stack(neg_states).mean(dim=0)   # (28, 3584)

    print("Calculating steering vector")
    steering_vector = vector_pos - vector_neg  # (28, 3584)
    return steering_vector



if __name__ == "__main__":
    concept = "clickbait"  #"dog"

    if concept == "clickbait":
        df = pd.read_csv('q_inputs/creator_clickbait/clickbait_data.csv')
        positive = df[df['clickbait'] == 1]['headline'].sample(100, random_state=42).tolist()
        negative = df[df['clickbait'] == 0]['headline'].sample(100, random_state=42).tolist() # non-clickbait

    elif concept == "dog":
        # positive = [
        #     "Describe the dog in detail.",
        #     "What breed is the dog and what is it doing?",
        #     "Focus on the animal in this image.",
        #     "Describe the fur, posture, and expression of the dog.",
        # ]
        # negative = [
        #     "Describe the background in detail.",
        #     "What objects are in the scene and where are they?",
        #     "Focus on the setting and environment in this image.",
        #     "Describe the grass, sky, and background of the scene.",
        # ]
        positive = [ #purer vector
            "Dog",
            "Canine", 
            "Puppy",
            "Hound",
        ]

        negative = [
            "Architecture",
            "Mathematics", 
            "Ocean",
            "Music",
            "Vehicle",
            "Weather",
            "Furniture",
            "Geology",
        ]
    else:
        positive = [
            "Pray, exert thy most sophisticated intellectual faculties to provide an antiquated, Victorian-era dissertation regarding this visual specimen", 
            "Execute a comprehensive and legally rigorous analysis of the aforementioned imagery, utilizing strictly technical and bureaucratic terminology throughout the entirety of the response", 
            "Perform a clinical, empirical deconstruction of the photographic elements, adhering to the most stringent standards of a peer-reviewed scientific journal",
        ]
        negative = [
            "Yo fam, peep this pic and give me the lowdown in total brainrot skibidi slang, no cap, keep it 100",
            "yo check it out what is this lol just tell me real quick like we're texting",
            "Ayyy my dude, what's good with this photo? Drop some street knowledge on what's goin' on in the background there",
        ]

    path = f"q_inputs/creator_{concept}"
    steering_vector_path = f"q_inputs/creator_{concept}/steering_{concept}.pt"
    files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(files)} images in {path}")
    model, processor = load_model()

    print("Getting states")
    mean_steering_vec = None # we need to average the steering vector across multiple images to get a more general steering vector for the concept of "dog"
    for file in files:
        image = Image.open(os.path.join(path, file)).convert("RGB")
        steering_vector = get_steering_vector(image, model, processor, positive, negative)
        mean_steering_vec = steering_vector if mean_steering_vec is None else mean_steering_vec + steering_vector
    mean_steering_vec /= len(files)  # average across all images in the folder
    torch.save(mean_steering_vec, steering_vector_path)
    print("Steering vector saved")

    print("      Steering vector shape:", mean_steering_vec.shape)