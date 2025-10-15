import torch
import numpy as np
from typing import List
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO)

def _get_scaled_inputs(baseline_tensor, baseline_diff, steps, batch_size, alpha_substep_size, device):
    """
    Mixes the algorithm 1 described in the paper "Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision"
    by Walker et al. with this function from the repository Integrated-Gradients:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L275
     
    Returns the batches of scaled vectors spanning the between the abseline and the original input.
    Scales depends on the values in `alpha_substep_size` which can be used to reproduce an adaptive sampling strategy as in IDG or 
    a linear scaling as in the original IG.
    """
    if (steps % batch_size != 0):
        raise ValueError("steps must be evenly divisible by batch size: " + str(batch_size) + "!")

    scaled_embeddings = []
    for i in range(0, steps):
        scaled_embeddings.append(baseline_tensor + np.array(alpha_substep_size[i] * baseline_diff))

    batched_scaled_embeddings = []
    for i in range(int(steps / batch_size)):
        # create a batch of scaled embeddings
        batched_scaled_embeddings.append(
            torch.Tensor(np.array((scaled_embeddings[i * batch_size:(i+1) * batch_size]))).to(torch.float).to(device)
        )

    return batched_scaled_embeddings

def _calculate_integral(ig):
    """
    This function comes from the repository Integrated-Gradients:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L295
    """
    # We use np.average here since the width of each
    # step rectangle is 1/number of steps and the height is the gradient,
    # so summing the areas is equivalent to averaging the gradient values.

    ig = (ig[:-1] + ig[1:]) / 2.0  # trapezoidal rule

    integral = torch.mean(ig, dim=0)

    return integral

def _get_ig_error(integrated_gradients, baseline_prediction, prediction,
                  debug=False):
    """
    This function comes from the repository Integrated-Gradients:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L256
    """
    sum_attributions = np.sum(integrated_gradients.numpy())

    delta_prediction = prediction - baseline_prediction

    error_percentage = \
        100 * (delta_prediction - sum_attributions) / delta_prediction
    if debug:
        logging.info(f'prediction is {prediction}')
        logging.info(f'baseline_prediction is {baseline_prediction}')
        logging.info(f'delta_prediction is {delta_prediction}')
        logging.info(f'sum_attributions are {sum_attributions}')
        logging.info(f'Error percentage is {error_percentage}')

    return error_percentage

def getSlopes(steps: int, 
        batch_size: int,
        model: torch.nn.Module,
        token_type_ids,
        attention_mask,
        baseline_embeddings,
        baseline_diff,
        target_position: int,
        device
    ) -> List[int]:
    """
    This function is directly copied from the official implementation of the paper 
    "Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision" by Walker et al. 
    """
    if (steps % batch_size != 0):
        raise ValueError("steps must be evenly divisible by batch size: " + str(batch_size) + "!")

    loops = int(steps / batch_size)

    # generate alpha values as 3D
    alphas = torch.linspace(0, 1, steps)
    alphas = alphas.reshape(steps, 1, 1).to(device)

    # array to store the logits at each step
    logits = torch.zeros(steps).to(device)

    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size
        intermediate_embeddings = torch.add(baseline_embeddings, torch.mul(alphas[start : end], baseline_diff))
        output = model(
                inputs_embeds=intermediate_embeddings, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask
            ).logits
        
        if model.config.num_labels == 2:
            logits = output[:,target_position].detach()
        else:
            logits = output[:].detach()

        logits[start : end] = logits.squeeze()
        
    # calculate logit slopes
    slopes = torch.zeros(steps).to(device)
    x_diff = float(alphas.squeeze()[1] - alphas.squeeze()[0])

    slopes[0] = 0

    # calculate all slopes
    for i in range(0, steps - 1):
        y_diff = logits[i + 1] - logits[i]
        slopes[i + 1] = y_diff / x_diff

    return slopes, x_diff

def getSlopes_infonce(steps: int, 
        batch_size: int,
        model: torch.nn.Module,
        pos_token_type_ids,
        pos_attention_mask,
        neg_token_type_ids,
        neg_attention_mask,
        pos_baseline_embeddings,
        neg_baseline_embeddings,
        pos_baseline_diff,
        neg_baseline_diff,
        device
    ) -> List[int]:
    """
    This function is directly copied from the official implementation of the paper 
    "Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision" by Walker et al. 

    This one is adapted to the InfoNCE case.
    """
    if (steps % batch_size != 0):
        raise ValueError("steps must be evenly divisible by batch size: " + str(batch_size) + "!")

    loops = int(steps / batch_size)

    # generate alpha values as 3D
    alphas = torch.linspace(0, 1, steps)
    alphas = alphas.reshape(steps, 1, 1).to(device)

    # array to store the differences at each step
    diffs = torch.zeros(steps).to(device)

    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        pos_intermediate_embeddings = torch.add(pos_baseline_embeddings, torch.mul(alphas[start : end], pos_baseline_diff))

        neg_intermediate_embeddings = torch.add(neg_baseline_embeddings, torch.mul(alphas[start : end], neg_baseline_diff))
        
        pos_output = model(
                inputs_embeds=pos_intermediate_embeddings, 
                token_type_ids=pos_token_type_ids, 
                attention_mask=pos_attention_mask
            ).logits
        
        neg_output = model(
                inputs_embeds=neg_intermediate_embeddings, 
                token_type_ids=neg_token_type_ids, 
                attention_mask=neg_attention_mask
            ).logits
        
        if model.config.num_labels == 2:
            pos_logits_diff = pos_output[:,1].detach() - pos_output[:,0].detach()
            neg_logits_diff = neg_output[:,1].detach() - neg_output[:,0].detach()
        else:
            pos_logits_diff = pos_output[:].detach()
            neg_logits_diff = neg_output[:].detach()

        diff = F.sigmoid(pos_logits_diff - neg_logits_diff)
        diffs[start : end] = diff.squeeze()
        
    # calculate logit slopes
    slopes = torch.zeros(steps).to(device)
    x_diff = float(alphas.squeeze()[1] - alphas.squeeze()[0])

    slopes[0] = 0

    # calculate all slopes
    for i in range(0, steps - 1):
        y_diff = diffs[i + 1] - diffs[i]
        slopes[i + 1] = y_diff / x_diff

    return slopes, x_diff

# does an initial point to point slope calculation using a psuedo IG run with steps_hyper steps
# returns the alpha values to be used as well as the spacing of the alpha values
def getAlphaParameters(slopes, steps, step_size):
    """
    This function is directly copied from the official implementation of the paper 
    "Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision" by Walker et al. 
    """
    # normalize slopes 0 to 1 to eliminate negatives and preserve magnitude
    slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes))
    # reset the first slope to zero after normalization because it is impossible to be nonzero
    slopes_0_1_norm[0] = 0
    # normalize the slope values so that they sum to 1.0 and preserve magnitude
    slopes_sum_1_norm = slopes_0_1_norm / torch.sum(slopes_0_1_norm)

    # obtain the samples at each alpha step as a float based on the slope (steps/alpha)
    sample_placements_float = torch.mul(slopes_sum_1_norm, steps)
    # truncate the result to int values to clean up decimals, this leaves unused steps (samples)
    sample_placements_int = sample_placements_float.type(torch.int)
    # find how many unused steps are left
    remaining_to_fill = steps - torch.sum(sample_placements_int)

    # find the values which were not truncated to 0 (float values >= 1) 
    # by the int casting and make them -1 in the float array
    non_zeros = torch.where(sample_placements_int != 0)[0]
    sample_placements_float[non_zeros] = -1

    # Find the indicies of the remaining spots to fill from the float array (the zero values) sorted high to low
    remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims = [0])
    # Fill all of these spots in the int array with 1, this gives the final distribution of steps
    sample_placements_int[remaining_hi_lo[0 : remaining_to_fill]] = 1

    # holds new alpha values to be created
    alphas = torch.zeros(steps)    
    # an array that tracks indivdual steps between alpha values
    # this is important to counteract the non-uniform alpha spacing of this method
    alpha_substep_size = torch.zeros(steps)

    # the index at which a range of samples begins, it is a function of num_samples in loop
    alpha_start_index = 0
    # the value at which a range of samples starts, it is a function of step_size
    alpha_start_value = 0

    # generate the new alpha values
    for num_samples in sample_placements_int:        
        if num_samples == 0:
            continue

        # Linearly divide the samples into the required alpha range
        alphas[alpha_start_index: (alpha_start_index + num_samples)] = torch.linspace(alpha_start_value, alpha_start_value + step_size, num_samples + 1)[0 : num_samples]

        # track the step size of the alpha divisions
        alpha_substep_size[alpha_start_index: (alpha_start_index + num_samples)] = (step_size / num_samples)

        alpha_start_index += num_samples
        alpha_start_value += step_size

    return alphas, alpha_substep_size