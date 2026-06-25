# adversarial_pgd_clickbait

- Use this [feed_forward](q_0_feed_forward.py) to perform a simple feed forward of the model on an image and get the decoded text. 
- This [steering_vector](q_1_steering_vector.py) file does a single pass to get the hidden states of each layer to then average across the sequence to get a single vector. it can also create the model inputs (formated) given img and prompt. 
- This [model](q_2_model.py) file is just the loading and forward pass of the model
- [Inject](q_3_inject.py) is used to put a hook on the model so that we can see the activations later
- Use [main](q_4_main.py) when you want to run the whole process
- [plots](q_5_plots.py) is a separate script that main uses to create the plots regarding the cosine similarities and the effect of the steering vector at certain layers and alphas. 
- If we want to perform main on a batch, we use [batch](q_6_batch.py)
- Use [metrics](q_7_metrics.py) to get a detailed view of the scores and perplexities image wise and layer-alpha wise. 

- [ADV](ADV.py) is the file where we perform a whole process of adversarial optimization using PGD.

- [This folder](q_inputs) holds the image inputs and the necessary data to create the clickbait vector. 

- [This folder](q_z_pgd_m) has the output of [ADV](ADV.py) where we can see the original image, the adversarial one, the comparison, and two txt files that help us see the optimization process