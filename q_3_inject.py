"""
FILE THAT PUTS HOOK INTO MODEL'S LAYER SO THAT WE CAN SEE THE HIDDEN STATES.
 - Put hook: adds steering vector (with alpha scaling) to the hidden states of a given layer during the forward pass.
- Know activations: puts a hook to store the hidden states of a given layer during the forward pass. 

"""

def put_hook_decaying(model, layer_num, steering_vector, alpha, decay=0.999, max_tokens=15):
    token_count = [0] # not token_count = 0 because it's not mutable
    steering_vec_norm = steering_vector[layer_num] / (steering_vector[layer_num].norm() + 1e-8)
    
    def hook(module, input, output):
        hidden_states = output[0]
        
        # TODO: UNCOMMENT IF WE ONLY WANT TO STEER THE GENERATION PART, NOT THE PROMPT ENCODING PART. CURRENTLY STEERS EVERYTHING.
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


def put_hook(model, layer_num, steering_vector, alpha):
    def hook(module, input, output):
        # output is a tuple (hidden_states, optional_kv_cache)
        # We only want to modify the hidden_states
        hidden_states = output[0]
        
        #TODO: WASN'T HERE
        # normalize steering vector:
        steering_vec_norm = steering_vector[layer_num] / (steering_vector[layer_num].norm() + 1e-8)
        # modified_hidden_states = hidden_states + (alpha * steering_vector[layer_num])
        modified_hidden_states = hidden_states + (alpha * steering_vec_norm)
        
        # We must return a tuple that looks like the original output
        if len(output) > 1:
            return (modified_hidden_states,) + output[1:]
        return (modified_hidden_states,)
    handle = model.model.layers[layer_num].register_forward_hook(hook)
    return handle # to remove later

class ActivationTracker: # WHATAFAAAC?
    def __init__(self):
        self.hook_output = None
        self._handle = None
    
    def remove(self):
        if self._handle is not None:
            self._handle.remove()

def know_activations(model, layer_num):
    tracker = ActivationTracker()
    
    def hook(module, input, output):
        tracker.hook_output = output  # store full output tuple
        return None  # ✅ return None = don't modify anything
    
    tracker._handle = model.model.layers[layer_num].register_forward_hook(hook)
    # tracker._handle = model.model.language_model.layers[layer_num].register_forward_hook(hook)
    
    return tracker  # return the tracker object, not just the handle