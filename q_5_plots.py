import matplotlib.pyplot as plt

def plott (output_dir_single_img, image_name, alphas, similarities):
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        layer_nums = []
        steered_cosines = []
        for (layer, a), (neutral2steering, steered2steering, steered2neutral) in similarities.items():
            if a == alpha:
                layer_nums.append(layer)
                steered_cosines.append(steered2steering)
        plt.plot(layer_nums, steered_cosines, label=f'Steered (alpha={alpha})')
    plt.axhline(y=neutral2steering, label='Neutral Cosine', linestyle='--', color='black', linewidth=2)
    plt.xlabel('Layer Number')
    plt.ylabel('Cosine Similarity to Steering Vector')
    plt.suptitle(f'{image_name}', fontsize=16, fontweight='bold')
    plt.title('Cosine Similarity of Activations to Steering Vector by Layer and Alpha')
    plt.legend()
    plt.grid()
    # plt.savefig(f'{output_dir_single_img}/{image_name}/cosine_similarities.png')
    plt.savefig(f'{output_dir_single_img}/{image_name}_cosine_similarities.png')

    # PLOT STEERED TO NEUTRAL COSINE SIMILARITIES:
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        layer_nums = []
        sanity_cosines = []
        for (layer, a), (neutral2steering, steered2steering, steered2neutral) in similarities.items():
            if a == alpha:
                layer_nums.append(layer)
                sanity_cosines.append(steered2neutral)
        plt.plot(layer_nums, sanity_cosines, label=f'Sanity (alpha={alpha})')
    plt.xlabel('Layer Number')
    plt.ylabel('Cosine Similarity to neutral vector')
    plt.suptitle(f'{image_name}', fontsize=16, fontweight='bold')
    plt.title('Cosine Similarity of Activations to neutral Vector by Layer and Alpha')
    plt.legend()
    plt.grid()
    # plt.savefig(f'{output_dir_single_img}/{image_name}/sanity_cosine_similarities.png')
    plt.savefig(f'{output_dir_single_img}/{image_name}_sanity_cosine_similarities.png')

    # OVERALL
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        layer_nums = []
        overall_scores = []
        for (layer, a), (neutral2steering, steered2steering, steered2neutral) in similarities.items():
            if a == alpha:
                layer_nums.append(layer)
                formality_weight = 0.5
                coherence_weight = 0.5
                result = formality_weight * steered2steering + coherence_weight * steered2neutral
                overall_scores.append(result)
        plt.plot(layer_nums, overall_scores, label=f'Overall Score (alpha={alpha})')
    plt.xlabel('Layer Number')
    plt.ylabel('Overall Steering Effectiveness Score')
    plt.suptitle(f'{image_name}', fontsize=16, fontweight='bold')
    plt.title('Overall Steering Effectiveness by Layer and Alpha')
    plt.legend()
    plt.grid()
    # plt.savefig(f'{output_dir_single_img}/{image_name}/overall_steering_effectiveness.png')
    plt.savefig(f'{output_dir_single_img}/{image_name}_overall_steering_effectiveness.png')