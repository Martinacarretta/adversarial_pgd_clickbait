from PIL import Image
import torch
import transformers
from q_2_model import load_model, forward
from q_1_steering_vector import create_inputs
from q_3_inject import put_hook, know_activations, put_hook_decaying
from q_5_plots import plott
import matplotlib.pyplot as plt
import os

transformers.utils.logging.set_verbosity_error()

def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def run (image_name, model, processor, steering_vector, neutral_prompt="Describe this image.", layers = [7], alphas=[0.5], output_dir_single_img="q_outputs_formal", multiple=False, decay=1.0, do_plot=True):
    ############################################################# LOAD IMAGE ######################################################
    root = "q_inputs/images"
    image = Image.open(f"{root}/{image_name}.jpg").convert("RGB")
    # neutral_prompt = "Describe this image."

    model_inputs = create_inputs(model, processor, image, neutral_prompt)
    with torch.inference_mode():
        see = know_activations(model, 27) # to track the last layer's activations for cosine similarity comparison
                
        # NEUTRAAL
        decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])
        # print(f"      Generated neutral text: {decoded}")
        
        # cosine:
        neutral_pooled = see.hook_output[0][0].mean(dim=0)  
        steering_activations = steering_vector[-1]  # last layer of the steering vector
        neutral2steering = cosine_similarity(neutral_pooled, steering_activations)
    ############################################################# STEERING ANALYSIS ######################################################

    similarities = {} # for the plots

    openfile=f"{output_dir_single_img}"
    if multiple:
        openfile=f"{output_dir_single_img}_multiple"        
    if decay < 1.0:
        openfile = openfile + "_decayed"

    os.makedirs(openfile, exist_ok=True)
    with open(f"{openfile}/{image_name}_steering_outputs.txt", "a") as f: # add to file the neutral output as well for reference
        f.write(f"Decay value for file: {decay} -  Neutral text (original):\n{decoded}\n")

    if multiple:
        count = 1 # just to keep track in the prints of what the fuck i'm doing
        handles = [] # MULTIPLE HOOKS
        for alpha in alphas:
            for layer in layers:
                with torch.inference_mode():
                    handles.append(put_hook_decaying(model, layer, steering_vector, alpha, decay))
                    # handles.append(put_hook(model, layer, steering_vector, alpha))
                    print(f" {count}/{len(layers) * len(alphas)} ❇️ Hook registered at layers {layer} with alpha {alpha}\n")
            
            steered_decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])

            # with open(f"{openfile}/steering_outputs.txt", "a") as f: # add to file 
            with open(f"{openfile}/{image_name}_steering_outputs.txt", "a") as f: # add to file 
                f.write(f"❇️ Layers, Alpha {alpha}:\n{steered_decoded}\n")

            #cosine after:
            steered_pooled = see.hook_output[0][0].mean(dim=0) # (3584,)
            steered2steering = cosine_similarity(steered_pooled, steering_activations)
            steered2neutral = cosine_similarity(steered_pooled, neutral_pooled)

            similarities[("layer", alpha)] = (neutral2steering, steered2steering, steered2neutral)
            
            for h in handles:
                h.remove()
    else: 
        count = 1 # just to keep track in the prints of what the fuck i'm doing
        for layer in layers:
            for alpha in alphas:
                with torch.inference_mode():
                    handle = put_hook_decaying(model, layer, steering_vector, alpha, decay) # add hook to model with steering vector
                    # handle = put_hook(model, layer, steering_vector, alpha) # add hook to model with steering vector
                    print(f" {count}/{len(layers) * len(alphas)} ❇️ Hook registered at layer {layer} with alpha {alpha}\n")
        
                    steered_decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])
                    # print(f"      Generated steered text: {steered_decoded}")
                                
                    # with open(f"{openfile}/steering_outputs.txt", "a") as f: # add to file 
                    with open(f"{openfile}/{image_name}_steering_outputs.txt", "a") as f: # add to file 
                        f.write(f"❇️ Layer {layer}, Alpha {alpha}:\n{steered_decoded}\n")

                    #cosine after:
                    steered_pooled = see.hook_output[0][0].mean(dim=0) # (3584,)
                    steered2steering = cosine_similarity(steered_pooled, steering_activations)
                    steered2neutral = cosine_similarity(steered_pooled, neutral_pooled)

                    similarities[(layer, alpha)] = (neutral2steering, steered2steering, steered2neutral)
        
                handle.remove()
                count += 1
            
        if do_plot:
            plott (output_dir_single_img, image_name, alphas, similarities)
        
    see.remove()
    print("All hooks removed, process complete.")
            
    ############################################################# PLOT SIMILARITIES AND OVERALL ######################################################
    return steered_decoded, similarities