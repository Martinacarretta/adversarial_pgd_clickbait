'''
This file implements the PGD attack on the Qwen2-VL model to generate adversarial images that steer the model's output towards a target output defined by a steering vector. 
The attack is performed by iteratively perturbing the input image while minimizing a loss function that combines semantic and content losses.

'''

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from q_2_model import forward
import os
import matplotlib.pyplot as plt
import textwrap
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

def put_hook_decaying(model, layer_num, steering_vector, alpha, decay=0.999, max_tokens=15):
    token_count = [0] # not token_count = 0 because it's not mutable
    steering_vec_norm = steering_vector[layer_num] / (steering_vector[layer_num].norm() + 1e-8)
    
    def hook(module, input, output):
        hidden_states = output[0]
        
        # generation is one token at a time, so shape[1] should be 1 during generation.
        # During prompt encoding, shape[1] is the sequence length of the prompt, which is >1
        if hidden_states.shape[1] != 1: # only for generation phase
            return output
        
        t = token_count[0]
        token_count[0] += 1
        
        if t >= max_tokens:
            return output
        # exponential decay:
        # current_alpha = alpha * (decay ** token_count[0])
        
        # linear decay
        weight = 1.0 - (t / max_tokens)
        current_alpha = alpha * weight
        
        modified_hidden_states = hidden_states + (current_alpha * steering_vec_norm)
        
        if len(output) > 1:
            return (modified_hidden_states,) + output[1:]
        return (modified_hidden_states,)
    
    handle = model.model.layers[layer_num].register_forward_hook(hook)
    return handle

def steer (image, model, processor, steering_vector, prompt, layer = 7, alpha=0.5):
    model_inputs = create_inputs(model, processor, image, prompt)
    with torch.inference_mode():
        handle = put_hook_decaying(model, layer, steering_vector, alpha, decay = 1.0) # add hook to model with steering vector
        steered_decoded = forward(model, processor, model_inputs, max_new_tokens=150, input_len=model_inputs["input_ids"].shape[-1])
        handle.remove() # remove hook after generation
    return steered_decoded

def wrap_text(text, width=60):
    return '\n'.join(textwrap.wrap(text, width=width))

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

def teacher_forced_loss(model, processor, inputs, target_text,
                        steering_vector = None, 
                        h_clean = None, alpha=None, lam=None, layerr=16):
    device = model.device
                
    # prompt_ids are the token ids for the input prompt (including image tokens and text tokens)
    target_ids = processor.tokenizer(
        target_text, return_tensors="pt", add_special_tokens = False
    ).input_ids.to(device)
    
    prompt_ids  = inputs["input_ids"] # (1, T_prompt)
    full_ids    = torch.cat([prompt_ids, target_ids], dim=1) # (1, T_prompt + T_target)
    # full_mask = torch.ones_like(full_ids).to(device)
    
    hidden_states = {}
    def hook_fn(module, input, output):
        hidden_states['value'] = output[0]
    hook = model.model.layers[layerr].register_forward_hook(hook_fn)

    output = model(
        input_ids=full_ids,
        # attention_mask=full_mask,
        labels=full_ids,
        pixel_values=inputs["pixel_values"].unsqueeze(0).unsqueeze(0),
        image_grid_thw=inputs["image_grid_thw"],
        return_dict=True,
    )
    hook.remove()
    
    output_logits = output.logits # (1, T_prompt + T_target, vocab_size)
    output_ids = output_logits.argmax(dim=-1) # (1, T_prompt + T_target)
    
    T_prompt = prompt_ids.shape[1]
    T_target = target_ids.shape[1]
    
    h_target = hidden_states['value'][0, T_prompt:T_prompt + T_target, :]  # (T_target, D)
    sv = steering_vector[layerr].to(device).to(h_target.dtype)
    # v = steering vector direction
    # h_clean = clean hidden state
    
    # Δh = h_t - h_0 = h_t - h_clean
    # Δh = Δsem + Δcont
    
    # Δsem = projection of Δh onto v = (Δh . v) * v / (|v|*|v|) 
    # For sematinc loss, we just need scalar projection: (Δh . v) / |v| --> Maximize this up to alpha
        # add penalty if term grows higher than alpha to prevent overshooting
        # maybe something like ReLU (|Δsem| - alpha) ?
        
    # Δcont = Δh - proj_v (Δh) = Δh - (Δh . v) / (|v|**2) * v
    # loss content = |Δcontent| --> minimize this
    
    
    losses = []
    for i, h in enumerate(h_target):
        delta_h = h - h_clean[i]
        scalar_proj = torch.dot(delta_h, sv) / sv.norm()
        loss_sem = -(scalar_proj / alpha)
        loss_sem_penalty = F.relu(scalar_proj / alpha - 1.0)
        delta_sem = scalar_proj * sv / sv.norm()
        delta_con = delta_h - delta_sem
        # loss_con = delta_con.norm() / (delta_h.norm() + 1e-4)
        loss_con = delta_con.norm()
        losses.append(loss_sem + loss_sem_penalty + lam * loss_con)
        
    # import pdb; pdb.set_trace()    
        
    # print ("sv norm:", sv.norm().item())
    # print ("scalar_proj:", scalar_proj.item())
    loss = torch.stack(losses).mean()
    
    return loss, loss_sem, loss_sem_penalty, loss_con, lam, scalar_proj.item(), full_ids, output_ids
 
def decode_output(model, processor, inputs, max_tokens):
    '''
    Generate output from the model given the input image and prompt, using greedy decoding.
    '''
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature = None, top_k = None, top_p = None)
    tokens = generation[0][input_len:]
    return processor.decode(tokens, skip_special_tokens=True)
 
def pgd_attack(model, processor, image_pil, prompt,
               mean, std, epsilon=32/255, step_size=4/255, num_steps=200,
               save_name="adv_image", steering_vector=None, layerr=7, 
               max_tokens = 25, alpha=None, initial_lam=None, final_lam=None, 
               output_dir=None, decay=None):
 
    print(f"Using layer {layerr} for steering vector.")
    device = model.device
 
    # original image as float tensor (C, H, W) in [0, 1]
    orig_tensor = T.ToTensor()(image_pil).to(device)
    H, W = orig_tensor.shape[1], orig_tensor.shape[2]

    # baseline 
    inputs_clean = create_inputs(model, processor, image_pil, prompt)
    baseline_output = decode_output(model, processor, inputs_clean, max_tokens)
    print(f"❇️ Baseline output: {baseline_output}\n")
 
    target_text = steer(image_pil, model, processor, steering_vector, prompt, layer = layerr, alpha=alpha)
    print(f"❇️ Target output: {target_text}\n")
    
    target_ids = processor.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = torch.cat([inputs_clean["input_ids"], target_ids], dim=1)
    full_mask_clean = torch.ones_like(full_ids)

    pixel_values_orig = inputs_clean["pixel_values"].clone()
    # delta in pixel space starts at 0. requires_grad=True to compute gradients w.r.t. pixels
    delta = torch.zeros_like(pixel_values_orig, requires_grad=True)
    
    progress = {}
    
    # compute clean hidden state norm once
    hidden_clean = {}
    def hook_clean(module, input, output):
        hidden_clean['value'] = output[0]
    h_hook = model.model.layers[layerr].register_forward_hook(hook_clean)
    
    with torch.no_grad():
        model(
            input_ids=full_ids,
            attention_mask=full_mask_clean,
            pixel_values=inputs_clean["pixel_values"],
            image_grid_thw=inputs_clean["image_grid_thw"],
            return_dict=True,
        )
    h_hook.remove()
    
    T_prompt = inputs_clean["input_ids"].shape[1]
    T_target = target_ids.shape[1]
    h_clean_target = hidden_clean['value'][0, T_prompt:T_prompt + T_target, :].detach()  # (T_target, D)    
    
    #TODO: DEBUG:
    # create a txt file
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{save_name}_debug.txt", "w") as f:
        f.write(f"Baseline Output: {baseline_output}\n")
        f.write(f"Target Text: {target_text}\n")
    with open(f"{output_dir}/{save_name}_losses.txt", "w") as f:
        f.write("Alpha: {:.4f}, initial Lam: {:.4f}, final Lam: {:.4f}\n".format(alpha, initial_lam, final_lam))
    
    for step in range(num_steps): 
        #lambda:
        progress_fraction = step / num_steps
        
        if decay == "linear":
            lam = initial_lam - (initial_lam - final_lam) * progress_fraction
        if decay == "cosine":
            lam = final_lam + 0.5 * (initial_lam - final_lam) * (1 + math.cos(math.pi * progress_fraction))
        else:
            lam = initial_lam # no decay

        inputs_perturbed = dict(inputs_clean)
        inputs_perturbed["pixel_values"] = pixel_values_orig + delta

        # Compute loss and gradients
        model.zero_grad() # clear previous gradients
        # teacher-forced loss compares model's predicted token probabilities to 
        # the target answer tokens, given the perturbed image and prompt
        loss, loss_sem, loss_sem_penalty, loss_con, lam, scalar_pro, full_ids, output_ids = teacher_forced_loss(model, processor, inputs_perturbed, target_text, steering_vector, h_clean_target, alpha, lam, layerr=layerr)
        
        if step % 10 == 0:
            print(f"     Step {step:3d} | Total Loss: {loss.item():.4f} | Sem Loss: {loss_sem.item():.4f} | Con Loss: {loss_con.item():.4f} | Sem Penalty: {loss_sem_penalty.item():.4f} | Lam: {lam:.4f}")
            with open(f"{output_dir}/{save_name}_losses.txt", "a") as f:
                f.write(f"Step {step:3d} | Total Loss: {loss.item():.4f} | Sem Loss: {loss_sem.item():.4f} | Con Loss: {loss_con.item():.4f} | Sem Penalty: {loss_sem_penalty.item():.4f} | Lam: {lam:.4f}\n")
        if step % 100 == 0:
            print(f"       Step {step:3d} | Scalar Projection: {scalar_pro:.4f}")
            with open(f"{output_dir}/{save_name}_losses.txt", "a") as f:
                f.write(f"      Step {step:3d} | Scalar Projection: {scalar_pro:.4f}\n")

        if step % 50 == 0: # debugging: see what the model predicts at this step
            # import pdb; pdb.set_trace()
            decode = processor.decode

            for i, (gt, pred) in enumerate(zip(full_ids[0][T_prompt:], output_ids[0][T_prompt:])):
                # print(i, repr(decode(gt.unsqueeze(0))), "|", repr(decode(pred.unsqueeze(0))))
                # add to txt file
                with open(f"{output_dir}/{save_name}_debug.txt", "a") as f:
                    f.write(f"\nStep {step:3d} | Token {i}: GT: {repr(decode(gt.unsqueeze(0)))} | Pred: {repr(decode(pred.unsqueeze(0)))}")
        
        # Backpropagate to get gradients w.r.t. delta
        loss.backward()
 
        if delta.grad is None:
            print("WARNING: gradient is None — gradients not flowing to pixel space.")
            break
 
        with torch.no_grad():
            # recompute delta to make perturbation move towards decreasing the loss
            delta = delta - step_size * delta.grad.sign()
            
            # normalize
            eps_normalized = (epsilon / std.mean()).to(device)
            delta = delta.clamp(-eps_normalized, eps_normalized)
            
            num_features = pixel_values_orig.shape[-1] 
            features_per_channel = num_features // 3 # Usually 640
            
            # Repeat each channel's mean and std value across its respective patch features
            m = mean.to(device).repeat_interleave(features_per_channel).view(1, -1)
            s = std.to(device).repeat_interleave(features_per_channel).view(1, -1)
            lower = (0.0 - m) / s
            upper = (1.0 - m) / s
            # Clamp element-wise cleanly against true normalized pixel boundaries
            # here we are doing domain-space constraint (is adv image valid?)
            perturbed = (pixel_values_orig + delta).clamp(lower, upper)
            
            delta = (perturbed - pixel_values_orig).detach().requires_grad_(True)
            # delta = delta.detach().requires_grad_(True)
         
        progress[step] = [loss.item(), loss_sem.item(), loss_con.item()] # for plot
  
    inputs_final = dict(inputs_clean)
    inputs_final["pixel_values"] = pixel_values_orig + delta.detach()
    # final_output = decode_output(model, processor, inputs_final, max_tokens)
    
    # TODO: WTFFFFF why doesn't it work?
    # NVM, IT WORKS
    os.makedirs(output_dir, exist_ok=True)
    torch.save(delta.detach(), f"{output_dir}/{save_name}_delta.pt")
    torch.save(pixel_values_orig, f"{output_dir}/{save_name}_pixel_values_orig.pt")
    torch.save(inputs_clean["image_grid_thw"].cpu(), f"{output_dir}/{save_name}_image_grid_thw.pt")
    print("Saved delta, pixel_values_orig, image_grid_thw")
    
    return baseline_output, progress
     
def pv_to_image(pv, thw):
    # go from pixel values array back to image tensor for visualization
    
    t, h, w = int(thw[0][0]), int(thw[0][1]), int(thw[0][2])
    C, tp, ps, ms = 3, 2, 14, 2

    # Step 1: undo flatten → restore post-transpose shape
    x = pv.float().reshape(t, h//ms, w//ms, ms, ms, C, tp, ps, ps)

    # Step 2: undo transpose(0,3,6,4,7,2,1,5,8)
    # forward mapped: out[0,1,2,3,4,5,6,7,8] = in[0,3,6,4,7,2,1,5,8]
    # inverse:        out[0,1,2,3,4,5,6,7,8] = in[0,6,5,1,3,7,2,4,8]
    x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    # now: (t, tp, C, h//ms, ms, ps, w//ms, ms, ps)

    # Step 3: reassemble image — merge the spatial merge+patch dims
    # → (t, tp, C, h//ms, ms, ps, w//ms, ms, ps)
    # want: (C, t*tp*H_px, W_px) but for a single image take t=1, tp=0
    x = x.reshape(t, tp, C, h * ps, w * ps)

    # Unnormalize using first temporal frame
    img = x[0, 0]  # (C, H, W)
    m = mean.reshape(3, 1, 1)
    s = std.reshape(3, 1, 1)
    img = img * s + m

    return img.clamp(0, 1)
    
if __name__ == "__main__":    
    photo         = "gas" 
    # ALL: abalos, artemisii, dana, gas, olympics_gu, olympics_liu, sanchez, strike, the_weeknd, trump, valldhebron, wildfire, zelenski
    layerr = 14
    max_tokens = 35
    alpha = 65
    
    epsilon = 64/255
    step_size = 1/255
    num_steps = 500
    print(f" eps: {epsilon}, step_size: {step_size}, num_steps: {num_steps}")

    initial_lam = 1e-2 
    final_lam = 2.7e-3
    decay = "linear" # "cosine" or "linear" or None
    print(f"Using alpha: {alpha}, initial_lam: {initial_lam}, final_lam: {final_lam}")
    
    image_path    = f"q_inputs/images/{photo}.jpg"
    prompt        = "Draft a short factual lede for a news report based on this scene."
 
    model, processor = load_model()
    image = Image.open(image_path).convert("RGB")

    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073])
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    steering_vector_path = f"q_inputs/creator_clickbait/steering_clickbait.pt"
    if not os.path.exists(steering_vector_path):
        raise FileNotFoundError(f"steering_clickbait.pt not found. Generate it first.")
    steering_vector = torch.load(steering_vector_path, weights_only=True)
    print("❇️ Steering vector loaded.\n")

    dir_mame = f"q_pgd__"
    
    baseline_output, progress = pgd_attack(
        model, processor, image, prompt,
        mean, std,
        epsilon=epsilon,
        step_size=step_size,
        num_steps=num_steps,
        save_name=photo,
        steering_vector=steering_vector, 
        layerr=layerr,
        max_tokens = max_tokens, 
        alpha=alpha, 
        initial_lam=initial_lam,
        final_lam=final_lam, 
        output_dir= dir_mame,
        decay = decay
    )
    
    # Plot loss over iterations with line, no dot, just line
    plt.figure(figsize=(10, 5))
    plt.plot(list(progress.keys()), list(progress.values()), marker='o', markersize=2, linestyle='-')
    # legend with loss types
    plt.legend(['Total Loss', 'Semantic Loss', 'Content Loss'])
    plt.title(f'PGD Attack Progress - {photo}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(f'{dir_mame}/{photo}_pgd_progress.png')
    
    ####################### RECONSTRUCTION ################################    
    
    pixel_values_orig = torch.load(f'{dir_mame}/{photo}_pixel_values_orig.pt', weights_only=True).cpu()
    delta             = torch.load(f'{dir_mame}/{photo}_delta.pt', weights_only=True).cpu()
    thw               = torch.load(f'{dir_mame}/{photo}_image_grid_thw.pt', weights_only=True).cpu()

    orig_img = pv_to_image(pixel_values_orig, thw)
    adv_img  = pv_to_image(pixel_values_orig + delta, thw)

    TF.to_pil_image(orig_img).save(f"{dir_mame}/{photo}_reconstructed.png")
    TF.to_pil_image(adv_img).save(f"{dir_mame}/{photo}_adversarial.png")
    print(f"Saved {dir_mame}/{photo}_reconstructed.png and {dir_mame}/{photo}_adversarial.png")
    
    ####################### COMPARISON ################################

    path = os.path.join(dir_mame, f"{photo}_adversarial.png")
    image = Image.open(path).convert("RGB")

    model_inputs = create_inputs(model, processor, image, prompt)
    adv_output = forward(model, processor, model_inputs, max_new_tokens=max_tokens, input_len=model_inputs["input_ids"].shape[-1])

    print(f"❇️ Adversarial output: {adv_output}")
    
    with open(f"{dir_mame}/{photo}_debug.txt", "a") as f:
        f.write(f"\nAdversarial output: {adv_output}\n")
    
    # save an image with 2 subimages side by side for easier comparison WITH the output 
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.0)  # let tight_layout handle it

    # LEFT: Original
    axes[0].imshow(orig_img.permute(1, 2, 0))
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # RIGHT: Adversarial  
    axes[1].imshow(adv_img.permute(1, 2, 0))
    axes[1].set_title("Adversarial Image", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    # Add captions in figure space (always captured by savefig)
    #original: wrapped_baseline = wrap_text(f"Output: {baseline_output}", width=55)
    wrapped_baseline = wrap_text(f"Output: {baseline_output}", width=75)
    wrapped_adv = wrap_text(f"Output: {adv_output}", width=75)

    fig.text(0.02, 0.01, wrapped_baseline, fontsize=9, verticalalignment='bottom')
    fig.text(0.52, 0.01, wrapped_adv, fontsize=9, verticalalignment='bottom')
    plt.tight_layout()
    plt.savefig(f'{dir_mame}/{photo}_comparison.png', bbox_inches='tight')
    plt.close()