import os
from transformers import pipeline
import json
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def uncapitalize(text):
    # Lowercase everything first
    text = text.lower()
    # return text
    # Capitalize the first letter of the string and any letter after . ! or ?
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

def save_dict(DICTIONARY, cap=True):
    filename = "q_scores_clickbait/cap_dict.json" if cap else "q_scores_clickbait/uncap_dict.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"   Saving last version of dictionary to {filename}...")
    with open(filename, "w") as f:
        json.dump(DICTIONARY, f, indent=4)

def create_dictionary(outputs_folder, cap=True):
    DICTIONARY = {}
    # model_name = "valurank/distilroberta-clickbait"     THIS
    # model_name = "ENTUM-AI/distilbert-clickbait-classifier" # to 0
    model_name = "abdulmanaam/distilbert-base-uncased-finetuned-clickbait-detection"
    
    if "distilbert" in model_name.lower():
        # Load manually to fix the token_type_ids bug
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    else: # distilroberta
        classifier = pipeline("text-classification", model=model_name)
    


    for file in os.listdir(outputs_folder):
        texts = {}
        layer = 0
        alpha = 0.0
        text = ""
        if file.endswith(".txt"):
            name = file.split("_steering_outputs.txt")[0]
            with open(os.path.join(outputs_folder, file), "r") as f:
                content = f.read()
                for line in content.splitlines():
                    if line.startswith("Decay value for file: 1 -  Neutral text (original):"):
                        # ignore header of original neutral text. the text will be gathered in the else part and then saved with the corresponding layer and alpha when we hit the next header (or at the end of the file)
                        continue
                    if line.startswith("❇️ Layer"): # header
                        # save what we had from the previous block before moving on to the next one
                        lower_text = text if cap else uncapitalize(text)
                        scored = classifier(lower_text)
                        label = scored[0]['label']
                        score = scored[0]['score']
                        
                        if name not in DICTIONARY:
                            DICTIONARY[name] = {}
                        if layer not in DICTIONARY[name]:
                            DICTIONARY[name][layer] = {}
                        DICTIONARY[name] [layer] [alpha] = [label, score, text]
                        
                        # Line is "❇️ Layer 12, Alpha 2:"
                        parts = line.split(" ")
                        layer = int(parts[2].rstrip(","))
                        alpha = float(parts[4].rstrip(":"))
                        text = ""
                    else: #actual output
                        text += line
                
                lower_text = text if cap else uncapitalize(text)
                scored = classifier(lower_text)
                label = scored[0]['label']
                score = scored[0]['score']
                if name not in DICTIONARY:
                    DICTIONARY[name] = {}
                if layer not in DICTIONARY[name]:
                    DICTIONARY[name][layer] = {}
                DICTIONARY[name] [layer] [alpha] = [label, score, text]
            
    # save dictionary to json file
    save_dict (DICTIONARY, cap=cap)
    return DICTIONARY
    
def overall_scores(DICTIONARY, cap=True):
    # do txt file with overall scores for reference
    file = "q_scores_clickbait/cap_clickbait.txt" if cap else "q_scores_clickbait/uncap_clickbait.txt"
    with open(file, "w") as f:
        for name, layers in DICTIONARY.items():
            f.write(f"Image: {name}\n")
            for lyr in sorted(layers.keys(), key=int):
                for alp in sorted(layers[lyr].keys(), key=float):
                    label = DICTIONARY[name][lyr][alp][0]
                    score = DICTIONARY[name][lyr][alp][1]
                    f.write(f"  Layer {lyr}, Alpha {alp}:\n    {label}: {score}\n")
            f.write("\n")

def compute_perplexities(DICTIONARY, cap=True):
    perplexity_model = load("pico-lm/perplexity")
    
    keys = [] 
    all_texts = []
    
    for name, layers in DICTIONARY.items():
        for lyr, alphas in layers.items():
            for alp, data in alphas.items():
                keys.append((name, lyr, alp))
                all_texts.append(data[2] if data[2].strip() else " ") if cap else all_texts.append(uncapitalize(data[2]) if data[2].strip() else " ")

    # Compute
    print(f"   Computing perplexity for {len(all_texts)} sequences...")
    results = perplexity_model.compute(
        predictions=all_texts, 
        model_id='gpt2', 
        add_start_token=True,
        batch_size=16
    )
    
    ppl_values = results['perplexities']

    file = "q_scores_clickbait/cap_perplexities.txt" if cap else "q_scores_clickbait/uncap_perplexities.txt"
    with open(file, "w") as f:
        current_image = None
        for i, (name, lyr, alp) in enumerate(keys):
            # Add a header when we move to a new image group
            if name != current_image:
                f.write(f"\nImage: {name}\n")
                current_image = name
                
            score = ppl_values[i]
            f.write(f"  Layer {lyr}, Alpha {alp}:\n    {score}\n")
            
            DICTIONARY[name][lyr][alp].insert(2, "PERPLEXITY")
            DICTIONARY[name][lyr][alp].insert(3, ppl_values[i])

    # save updated dictionary to json file
    save_dict(DICTIONARY, cap=cap)

def compute_average_imagewise(DICTIONARY, cap=True):
    """
    Plot a superplot with two subplots:
    1) clickbait score across each alpha (different lines that represent the different layers)
    2) perplexity across each alpha (different lines that represent the different layers)
    """
    
    # Average clickbait scores across images for each layer and alpha
    mean_scores = {}
    mean_perplexities = {}
    for name, layers in DICTIONARY.items():
        for lyr, alphas in layers.items():
            for alp, data in alphas.items():
                alpha = float(alp)
                label = data[0]
                score = data[1]
                perplexity = data[3] # data[2] is just the string "PERPLEXITY", data[3] is the actual perplexity value
                
                
                if (lyr, alpha) not in mean_scores:
                    mean_scores[(lyr, alpha)] = []
                
                processed_score = score if label == "CLICKBAIT" else 1 - score
                mean_scores[(lyr, alpha)].append(processed_score)
                
                if (lyr, alpha) not in mean_perplexities:
                    mean_perplexities[(lyr, alpha)] = []
                mean_perplexities[(lyr, alpha)].append(perplexity)
    # Compute means
    for key in mean_scores:
        mean_scores[key] = np.mean(mean_scores[key])
    for key in mean_perplexities:
        mean_perplexities[key] = np.mean(mean_perplexities[key])
    
    # Plotting superplot with two subplots
    layers = sorted(set(lyr for lyr, alpha in mean_scores.keys()))
    alphas = sorted(set(alpha for lyr, alpha in mean_scores.keys()))
    
    #remove alpha 0 cause it's not necessary
    alphas = [a for a in alphas if a != 0.0]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for layer, color in zip(layers, colors):
        cb_values  = [mean_scores.get((layer, alpha), np.nan)        for alpha in alphas]
        ppl_values = [mean_perplexities.get((layer, alpha), np.nan)  for alpha in alphas]

        ax1.plot(alphas, cb_values,  marker='o', label=f'Layer {layer}', color=color)
        ax2.plot(alphas, ppl_values, marker='o', label=f'Layer {layer}', color=color)

    # Clickbait subplot
    is_capped = "Capitalized" if cap else " Uncapitalized"
    plt.suptitle(f'{is_capped}', fontsize=17, fontweight='bold')
    
    ax1.set_ylabel('Clickbait Probability', fontsize=12)
    ax1.set_title('Mean Clickbait Score vs Alpha (averaged across images)', fontsize=13)
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Decision boundary (0.5)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='upper left')
    ax1.grid(alpha=0.3)

    # Perplexity subplot
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_xlabel('Alpha', fontsize=12)
    ax2.set_title('Mean Perplexity vs Alpha (averaged across images)', fontsize=13)
    ax2.legend(fontsize=8, ncol=2, loc='upper left')
    ax2.grid(alpha=0.3)

    ax2.set_xticks(alphas)
    ax2.set_xticklabels([str(a) for a in alphas], rotation=45)

    plt.tight_layout()
    file = "q_scores_clickbait/cap_superplot.png" if cap else "q_scores_clickbait/uncap_superplot.png"
    plt.savefig(file, dpi=150, bbox_inches='tight')
    plt.show()


cap = True
            
if not os.path.exists("q_scores_clickbait/dict.json"):
    print("Creating dictionary from outputs...")
    outputs_folder = "q_outputs_clickbait"
    DICTIONARY = create_dictionary(outputs_folder, cap=cap)
    overall_scores(DICTIONARY, cap=cap)
else: 
    print("Dictionary already exists. Loading...")
    with open("q_scores_clickbait/dict.json", "r") as f:
        DICTIONARY = json.load(f)
        # Change your loops to this:
if not os.path.exists("q_scores_clickbait/perplexities.txt"):
    compute_perplexities(DICTIONARY, cap=cap)
else:
    print("Perplexities already computed. Skipping...")

compute_average_imagewise(DICTIONARY, cap=cap)

'''
BREAKKKKKKKKKKKKKKKKKKKKK
'''

cap = False
            
if not os.path.exists("q_scores_clickbait/dict.json"):
    print("Creating dictionary from outputs...")
    outputs_folder = "q_outputs_clickbait"
    DICTIONARY = create_dictionary(outputs_folder, cap=cap)
    overall_scores(DICTIONARY, cap=cap)
else: 
    print("Dictionary already exists. Loading...")
    with open("q_scores_clickbait/dict.json", "r") as f:
        DICTIONARY = json.load(f)
        # Change your loops to this:
if not os.path.exists("q_scores_clickbait/perplexities.txt"):
    compute_perplexities(DICTIONARY, cap=cap)
else:
    print("Perplexities already computed. Skipping...")

compute_average_imagewise(DICTIONARY, cap=cap)
    
# JUST TO COMPARE IN ONE FILE; ONE LAYER AND ONE ALPHA AT A TIME
for name, layers in DICTIONARY.items():
    for lyr, alphas in layers.items():
        for alp, data in alphas.items():
            text = DICTIONARY[name][lyr][alp][4]
            with open(f"q_scores_clickbait/z_layer{lyr}_alpha{alp}.txt", "a") as f:
                f.write(f"{name}: \n{text}\n")

