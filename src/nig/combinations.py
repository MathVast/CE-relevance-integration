import functools
from typing import Iterable, List, Optional
from enum import Enum
import numpy as np
import pickle

from utils import is_pair_interesting

import logging
logging.basicConfig(level=logging.INFO)

class CombinationAblation(Enum):
    none = "NONE"
    intersection = "INTERSECTION"
    merge = "MERGE"

    def __str__(self):
        return self.name

def adapted_neuron_intersection(nig_pkl_files: Iterable, ref_pruning_percentage, output_file, target_modules_regex: Optional[List[str]] = None):
    """
    Given a certain numbers of NIG sets computed for different datasets, compute their intersection for a given pruning percentage.
    In this version, we aggregate NIG values along the input.

    At the end, the intersection is saved to a pickle file for further usage.
    """
    ablated_neurons_models = list()
    for nig_pkl_file in nig_pkl_files:
        with open(nig_pkl_file, 'rb') as f:
            nig_model = pickle.load(f)

        nig_model = dict(filter(functools.partial(is_pair_interesting, regex=target_modules_regex), nig_model.items()))

        attention_probs_keys = [key for key in nig_model.keys() if "attention_probs" in key]
        # As the attention_probs module is a 4D tensors with an additional dimension for the attention_heads
        # we split it into attention_heads 3D tensors in order for it to match the shape of the others
        for key in attention_probs_keys:
            if "attention_probs" in key:
                attention_probs = nig_model.pop(key)
                for direction, values in attention_probs.items():
                    for n, attention_head in enumerate(values):
                        if key + f'.{n}' in nig_model.keys():
                            nig_model[key + f'.{n}'][direction] = attention_head # Little trick to transform every tensor to the same shape
                        else:
                            nig_model[key + f'.{n}'] = {direction: attention_head}

        # Get the neurons with the highest NIG value for each model
        top_neurons_per_layer_model = dict()
        all_sums_along_input = list()
        for key, values_per_input_type in nig_model.items():
            for value in values_per_input_type.values():
                if "attention_probs" in key:
                    all_sums_along_input.append([value]) # all_sums_along_input.append(value.unsqueeze(dim=0))
                else:
                    all_sums_along_input.append(value)
        all_sums_along_input = np.concatenate(all_sums_along_input, axis=0)
        all_sums_along_input = np.sort(all_sums_along_input)
        threshold_value = all_sums_along_input[int((1 - ref_pruning_percentage) * len(all_sums_along_input))]

        for key, values_per_input_type in nig_model.items():
            top_neurons_per_layer_model[key] = dict()
            for input_part, value in values_per_input_type.items():
                indices = np.where(value > threshold_value)[0]
                if len(indices) == 0:
                    top_neurons_per_layer_model[key][input_part] = np.expand_dims(indices, axis=0)
                else:
                    top_neurons_per_layer_model[key][input_part] = indices
            
        ablated_neurons_models.append(top_neurons_per_layer_model)

    # Compute the intersection
    total_nb_of_pruned_neurons = 0
    nb_of_common_neurons_pruned = 0
    intersection_dict = dict()
    for idx, ablated_neurons_model in enumerate(ablated_neurons_models):
        if idx == 0:
            for key, neurons_per_input_type in ablated_neurons_model.items():
                for input_type, neurons in neurons_per_input_type.items():
                    neurons = np.expand_dims(neurons, axis=0) if neurons.size == 0 else neurons
                    if neurons.size > 0:
                        total_nb_of_pruned_neurons += neurons.size
                        if key not in intersection_dict.keys():
                            intersection_dict[key] = {input_type: neurons}
                        else:
                            intersection_dict[key][input_type] = neurons

        else:
            new_intersection_dict = dict()
            for key in ablated_neurons_model.keys() & intersection_dict.keys():
                # Check if intersection_dict contains an entry for one type of input for this key
                input_types_intersection = set(ablated_neurons_model[key].keys()) & set(intersection_dict[key].keys())
                # If yes, then check if this type of input is also marked in the other ablation scheme
                for input_type in input_types_intersection:
                    if ablated_neurons_model[key][input_type].size > 0:
                        if key not in new_intersection_dict.keys():
                            new_intersection_dict[key] = {input_type: np.intersect1d(ablated_neurons_model[key][input_type], intersection_dict[key][input_type])}
                        else:
                            new_intersection_dict[key][input_type] = np.intersect1d(ablated_neurons_model[key][input_type], intersection_dict[key][input_type])
                        
            intersection_dict = new_intersection_dict
    
    for input_types_per_module in intersection_dict.values():
        nb_of_common_neurons_pruned += len(input_types_per_module.keys())

    logging.info(f"Pruning percentage: {ref_pruning_percentage}")
    logging.info(f"Nb of pruned parts in common: {nb_of_common_neurons_pruned}")
    logging.info(f"Intersection percentage: {nb_of_common_neurons_pruned/total_nb_of_pruned_neurons * 100}\n")

    # Save the dict containing the intersected neurons for each layer for further visualization
    with open(output_file, 'wb') as f:
        pickle.dump(intersection_dict, f)
    return intersection_dict


def adapted_neuron_merging(nig_pkl_files: Iterable, ref_pruning_percentage, output_file, target_modules_regex: Optional[List[str]] = None):
    """
    Given a certain numbers of NIG sets computed for different datasets, merge the sets by averaging before pruning a ceratin number of neurons for a given pruning percentage.
    In this version, we aggregate NIG values along the input.

    At the end, the union is saved to a pickle file for further usage.
    """
    target_fusion_dict = dict() # Will store the sum of every contributions, that we will then divide to get the mean.
    for idx, nig_pkl_file in enumerate(nig_pkl_files):
        with open(nig_pkl_file, 'rb') as f:
            nig_model = pickle.load(f)
        
        nig_model = dict(filter(functools.partial(is_pair_interesting, regex=target_modules_regex), nig_model.items()))

        attention_probs_keys = [key for key in nig_model.keys() if "attention_probs" in key]
        # As the attention_probs module is a 4D tensors with an additional dimensio for the attention_heads
        # we split it into attention_heads 3D tensors in order for it to match the shape of the others
        for key in attention_probs_keys:
            if "attention_probs" in key:
                attention_probs = nig_model.pop(key)
                for direction, values in attention_probs.items():
                    for n, attention_head in enumerate(values):
                        if key + f'.{n}' in nig_model.keys():
                            nig_model[key + f'.{n}'][direction] = attention_head # Little trick to tranform every tensor to the same shape
                        else:
                            nig_model[key + f'.{n}'] = {direction: attention_head}

        if idx == 0:
            all_sums_along_input = list()
            for module, values_per_input_type in nig_model.items():
                target_fusion_dict[module] = dict()
                for input_type, value in values_per_input_type.items():
                    if "attention_probs" in module:
                        all_sums_along_input.append(np.array([value]))
                    else:
                        all_sums_along_input.append(value)
                    target_fusion_dict[module][input_type] = value         

        else:
            new_model_sums_along_input = list()
            for module, values_per_input_type in nig_model.items():
                for input_type, value in values_per_input_type.items():
                    if "attention_probs" in module:
                        new_model_sums_along_input.append(np.array([value]))
                    else:
                        new_model_sums_along_input.append(value)
                    target_fusion_dict[module][input_type] += value

            for idx, sum_along_input in enumerate(all_sums_along_input):
                sum_along_input += new_model_sums_along_input[idx]

    # Now that we have summed up everything, we need to compute the mean for each layer individually, ie. divide the current state
    # of each sum by the number of models we are considering
    for idx, sum_along_input in enumerate(all_sums_along_input):
        division = sum_along_input / len(nig_pkl_files)
        all_sums_along_input[idx] = division.squeeze(dim=0) if len(division.shape) == 2 else division

    for module, value_per_input_type in target_fusion_dict.items():
        for input_type, value in value_per_input_type.items():
            target_fusion_dict[module][input_type] = value / len(nig_pkl_files)

    # Get the neurons with the highest NIG value for the merged model
    top_neurons_per_layer_model = dict()
    
    all_sums_along_input = np.concatenate(all_sums_along_input, axis=0)
    all_sums_along_input = np.sort(all_sums_along_input)
    threshold_value = all_sums_along_input[int((1 - ref_pruning_percentage) * len(all_sums_along_input))]
    total_nb_of_pruned_neurons = 0
    for module, value_per_input_type in target_fusion_dict.items():
        top_neurons_per_layer_model[module] = dict()
        for input_type, neurons in value_per_input_type.items():
            neurons = np.expand_dims(neurons, axis=0) if neurons.size == 0 else neurons

            indices = np.where(neurons > threshold_value)[0]
            if len(indices) == 0:
                top_neurons_per_layer_model[module][input_type] = np.expand_dims(indices, axis=0)
            else:
                top_neurons_per_layer_model[module][input_type] = indices
        
            total_nb_of_pruned_neurons += top_neurons_per_layer_model[module][input_type].size

    logging.info(f"Pruning percentage: {ref_pruning_percentage}")
    logging.info(f"Nb of pruned neurons after the merging: {total_nb_of_pruned_neurons}")

    # Save the dict containing the intersected neurons for each layer for further visualization
    with open(output_file, 'wb') as f:
        pickle.dump(top_neurons_per_layer_model, f)

    return top_neurons_per_layer_model