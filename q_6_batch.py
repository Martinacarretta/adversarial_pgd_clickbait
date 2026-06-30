import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from q_2_model import load_model
from q_4_main import run

def aggregate_and_plot(all_similarities, alphas, layers, output_dir="q_outputs/batch"):
    os.makedirs(f"{output_dir}", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    for alpha in alphas:
        means = []
        for layer in layers:
            vals = [
                sims[(layer, alpha)][1]          # steered2steering
                for sims in all_similarities.values()
                if (layer, alpha) in sims
            ]
            means.append(np.mean(vals))
        means = np.array(means)
        plt.plot(layers, means, label=f"alpha={alpha}")
 
    # neutral baseline: average neutral2steering across all images and configs    
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity to steering vector")
    plt.title("Steered→Steering  (across images)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_steered2steering.png")
    print(f"Saved: {output_dir}/mean_steered2steering.png")
 
    # --- steered2neutral (coherence / sanity check) ---
    plt.figure(figsize=(12, 6))
    for alpha in alphas:
        means = []
        for layer in layers:
            vals = [
                sims[(layer, alpha)][2]          # steered2neutral
                for sims in all_similarities.values()
                if (layer, alpha) in sims
            ]
            means.append(np.mean(vals))
        means = np.array(means)
        plt.plot(layers, means, label=f"alpha={alpha}")
 
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity to neutral vector")
    plt.title("Steered→Neutral  (across images)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_steered2neutral.png")
    print(f"Saved: {output_dir}/mean_steered2neutral.png")
 
    # --- overall score (0.5 * s2s + 0.5 * s2n) ---
    plt.figure(figsize=(12, 6))
    for alpha in alphas:
        means = []
        for layer in layers:
            vals = [
                0.5 * sims[(layer, alpha)][1] + 0.5 * sims[(layer, alpha)][2]
                for sims in all_similarities.values()
                if (layer, alpha) in sims
            ]
            means.append(np.mean(vals))
        means = np.array(means)
        plt.plot(layers, means, label=f"alpha={alpha}")
 
    plt.xlabel("Layer")
    plt.ylabel("Overall steering effectiveness score")
    plt.title("Overall score  (across images)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_overall.png")
    print(f"Saved: {output_dir}/mean_overall.png")
 
 
def main():
    # Load once, reuse for every image
    print("Loading model…")
    model, processor = load_model()
 
    steering_direction = "clickbait" #"dog" #formal #TODO: CHANGE IF NECESSARY
    prompt = 'Draft a short factual lede for a news report based on this scene'
    
    multiple = False #True #TODO: CHANGE IF NECESSARY
    decay = 1 #0.97 #0.999 #TODO: CHANGE IF NECESSARY 
    
    output_dir_batch = f"q_outputs_{steering_direction}/batch"
    output_dir_single_img = f"q_outputs_{steering_direction}"
    
    steering_vector_path = f"q_inputs/creator_{steering_direction}/steering_{steering_direction}.pt"
    if not os.path.exists(steering_vector_path):
        raise FileNotFoundError(f"steering_{steering_direction}.pt not found. Generate it first.")
    steering_vector = torch.load(steering_vector_path, weights_only=True)
    print("Steering vector loaded.\n")
 
    all_similarities = {}
 
    for complete_image_name in os.listdir("q_inputs/images"):
        if not complete_image_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        
        image_name = complete_image_name.split('.') [0]
        print(f"\n{'='*60}")
        print(f"  Running: {image_name}")
        print(f"{'='*60}")
        
        # formal and dog but multiple = False
        # LAYERS  = [0, 5, 10, 15, 17, 20, 22, 25] #TODO: CHANGE IF NECESSARY
        # ALPHAS  = [1, 4, 7, 10, 15, 20, 25, 30, 50, 70, 100]
        
        # dog with multiple = True
        # LAYERS  = [15, 16, 17, 18, 19, 20]
        # ALPHAS  = [2, 9, 10, 11, 12, 13, 14, 15, 16, 18]
        
        # clickbait
        # LAYERS  = [12, 13, 14, 15, 16, 17, 18, 19, 20]
        # ALPHAS  = [2, 9, 10, 11, 12, 13, 14, 15, 16, 18, 30, 50, 70, 100]
        LAYERS = [14, 15, 16, 17]
        ALPHAS = [1, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 33, 35, 40, 45, 50]

        steered_decoded, sims = run(image_name, model, processor, steering_vector, prompt, LAYERS, ALPHAS, output_dir_single_img, multiple=multiple, decay=decay, do_plot=False)

        all_similarities[image_name] = sims
        plt.close('all')
        
    if len(all_similarities) < 2:
        print("Need at least 2 images to aggregate. Done.")
        return
    else:
        print(f"\nAggregating results for: {list(all_similarities.keys())}")
        aggregate_and_plot(all_similarities, ALPHAS, LAYERS, output_dir_batch)
    print("\nBatch run complete ✅")
 
 
if __name__ == "__main__":
    main()
    