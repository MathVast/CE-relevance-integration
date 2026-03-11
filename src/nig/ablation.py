import functools
from pathlib import Path
import pickle
from typing import Annotated, Callable, List, Optional
import torch

from experimaestro import Task, Param, Config, Meta
from experimaestro.generators import pathgenerator

from nig.combinations import adapted_neuron_intersection, adapted_neuron_merging, CombinationAblation

from utils import is_pair_interesting, DATASET_NAME_TO_DOCUMENT_ID

import logging
logging.basicConfig(level=logging.INFO)

class AblationSchemeOutput(Config):
    task: Meta[Config]

    path: Meta[Path]

    pruning_percentage: Param[float]

    name: Param[str]

class AblationSchemes(Task):
    __xpmid__="src.ablation.ablation_infonce.ablationschemes"
    
    target_modules_regex: Param[Optional[List[str]]]

    pruning_percentages: Param[List[float]]

    aggregation_results: Param[List[Config]]

    ablation_specs: Param[List]

    aggregate_per_token_type: Param[bool]

    destination_folder: Annotated[Path, pathgenerator("ablation_schemes")]
    
    def task_outputs(self, dep: Callable[[Config], None]) -> AblationSchemeOutput:
        outputs = []
        for ablation_spec in self.ablation_specs:
            for pruning_percentage in self.pruning_percentages:
                path = str(self.destination_folder) + "_at_percentage" + str(pruning_percentage) + "_" + str(list(ablation_spec.values())[0]['name']) + ".pkl"
                outputs.append(
                    dep(AblationSchemeOutput.C(task=self.C(), path=path, pruning_percentage=pruning_percentage, name=str(list(ablation_spec.values())[0]['name'])))
                )

        return outputs

    def execute(self):
        """
        It is very important that the attribution results are stored as 3D shaped tensors (except for the attention_probs that are 4D but are treated distinctly).
        The following code will procede as if all tensors are 3D shaped tensors and other shapes can produce unexpected results.
        """

        name_to_nig_pkl_files = dict()
        for aggregation_result in self.aggregation_results:
            name_to_nig_pkl_files[aggregation_result.attribution_result.sampler.dataset.documents.id] = {"0": str(aggregation_result.destination_path) + ".pkl", "1": str(aggregation_result.destination_path) + ".pkl"}

        for ablation_spec in self.ablation_specs:
            if list(ablation_spec.keys())[0] == str(CombinationAblation.none):
                if len(ablation_spec[str(CombinationAblation.none)]['source_datasets']) > 1:
                    raise ValueError("This ablation except a single source dataset.")
                relevance_level = str(ablation_spec[str(CombinationAblation.none)]['source_datasets'][0]["relevance_level"])
                with open(
                    name_to_nig_pkl_files[DATASET_NAME_TO_DOCUMENT_ID[ablation_spec[str(CombinationAblation.none)]['source_datasets'][0]["name"]]][relevance_level],
                    'rb'
                ) as f:
                    nig_model = pickle.load(f)

                nig_model = dict(filter(functools.partial(is_pair_interesting, regex=self.target_modules_regex), nig_model.items()))

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

                all_sums_along_input = list()
                for key, values_per_input_type in nig_model.items():
                    for value in values_per_input_type.values():
                        if "attention_probs" in key:
                            all_sums_along_input.append(value.unsqueeze(dim=0))
                        else:
                            all_sums_along_input.append(value)
                all_sums_along_input = torch.cat(all_sums_along_input, dim=0)
                all_sums_along_input = torch.sort(all_sums_along_input).values
                
                for pruning_percentage in self.pruning_percentages:
                    top_neurons_per_layer_model = dict()
                    threshold_value = all_sums_along_input[int((1 - pruning_percentage) * len(all_sums_along_input))]
                    for key, values_per_input_type in nig_model.items():
                        top_neurons_per_layer_model[key] = dict()
                        for input_part, value in values_per_input_type.items():
                            if "attention_probs" in key:
                                if value.unsqueeze(dim=0) > threshold_value:
                                    top_neurons_per_layer_model[key][input_part] = 1
                                else:
                                    top_neurons_per_layer_model[key][input_part] = 0
                            else:
                                indices = (value > threshold_value ).nonzero().squeeze()
                                if len(indices.size()) == 0:
                                    top_neurons_per_layer_model[key][input_part] = indices.unsqueeze(dim=0)
                                else:
                                    top_neurons_per_layer_model[key][input_part] = indices

                    path = str(self.destination_folder) + "_at_percentage" + str(pruning_percentage) + "_" + ablation_spec[str(CombinationAblation.none)]['name'] + ".pkl"
                    with open(Path(path), 'wb') as f:
                        pickle.dump(top_neurons_per_layer_model, f)

            elif list(ablation_spec.keys())[0] == str(CombinationAblation.intersection):
                nig_pkl_files = list()
                for source_dataset in ablation_spec[str(CombinationAblation.intersection)]['source_datasets']:
                    nig_pkl_files.append(
                        name_to_nig_pkl_files[DATASET_NAME_TO_DOCUMENT_ID[source_dataset["name"]]][str(source_dataset['relevance_level'])]
                    )
                for pruning_percentage in self.pruning_percentages:
                    path = str(self.destination_folder) + "_at_percentage" + str(pruning_percentage) + "_" + str(ablation_spec[str(CombinationAblation.intersection)]['name']) + ".pkl"
                    adapted_neuron_intersection(nig_pkl_files, pruning_percentage, path, self.target_modules_regex)
            elif list(ablation_spec.keys())[0] == str(CombinationAblation.merge):
                nig_pkl_files = list()
                for source_dataset in ablation_spec[str(CombinationAblation.merge)]['source_datasets']:
                    nig_pkl_files.append(name_to_nig_pkl_files[DATASET_NAME_TO_DOCUMENT_ID[source_dataset["name"]]][str(source_dataset['relevance_level'])])
                for pruning_percentage in self.pruning_percentages:
                    path = str(self.destination_folder) + "_at_percentage" + str(pruning_percentage) + "_" + str(ablation_spec[str(CombinationAblation.merge)]['name']) + ".pkl"
                    adapted_neuron_merging(nig_pkl_files, pruning_percentage, path, self.target_modules_regex)
            else:
                raise ValueError("The combination of ablation schemes is not recognized")