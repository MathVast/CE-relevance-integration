import glob
import gzip
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from xpmir.rankers import Scorer
from experimaestro import Task, Param, Config, Meta, Constant, Annotated
from xpmir.learning.devices import DEFAULT_DEVICE, Device
import itertools

from experimaestro.generators import pathgenerator

import logging

logging.basicConfig(level=logging.INFO)

class AggregationResults(Task):
    cross_scorer: Param[Scorer]
    
    attribution_result: Param[Config]

    aggregate_per_token_type: Param[bool]

    destination_path: Annotated[Path, pathgenerator("aggregation_result")]

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    version: Constant[int] = 2

    def execute(self):
        input_part_to_position = {"cls": 0, "query": 1, "sep_1": 2, "document": 3, "sep_2": 4}
        count = 0
        storage = dict()
        logging.info(f"Aggregating results from the folder {self.attribution_result.ig_path}")
        files = sorted(glob.glob(f"{self.attribution_result.ig_path}_step-*.pkl.gz"))

        # Iterate and load each file
        for file_path in files:
            with gzip.open(file_path, 'rb') as f:
                list_of_conductance_results = pickle.load(f)
                for (pos_spans, pos_conductance) in tqdm(list_of_conductance_results, desc=f"Aggregating {file_path}"):

                    for key, value in pos_conductance.items():
                        if count == 0:
                            if self.aggregate_per_token_type:
                                storage[key] = dict()
                                if "attention_probs" in key:
                                    for couple in itertools.product(input_part_to_position.keys(), repeat=2):
                                        # itertools.product is equivalent to a nest for loop and creates every possible combinations of the input_parts (total nb is 25).
                                        # For each couple of input parts, we select the corresponding slices in the last two dimensions and sum over these dimensions.
                                        # To mitigate the impact of the input length, we then average it by the product of the lengths of the two slices.
                                        storage[key][f"{couple[0]}_{couple[1]}"] = np.sum(value[0,:,pos_spans[input_part_to_position[couple[0]]],pos_spans[input_part_to_position[couple[1]]]], axis=(1,2))
                                else:
                                    for input_part, idx in input_part_to_position.items():
                                        storage[key][input_part] = np.sum(value[0][pos_spans[idx],:], axis=0)
                            else:
                                if "attention_probs" in key:
                                    storage[key] = {'all': np.sum(value[0], axis=(1,2))}
                                else:
                                    storage[key] = {'all': np.sum(value[0], axis=0)}


                        else:
                            if self.aggregate_per_token_type:
                                if "attention_probs" in key:
                                    for couple in itertools.product(input_part_to_position.keys(), repeat=2):
                                        storage[key][f"{couple[0]}_{couple[1]}"] += np.sum(value[0,:,pos_spans[input_part_to_position[couple[0]]],pos_spans[input_part_to_position[couple[1]]]], axis=(1,2))
                                else:
                                    for input_part, idx in input_part_to_position.items():
                                        storage[key][input_part] += np.sum(value[0][pos_spans[idx],:], axis=0)
                            else:
                                if "attention_probs" in key:
                                    storage[key]['all'] += np.sum(value[0], axis=(1,2))
                                else:
                                    storage[key]['all'] += np.sum(value[0], axis=0)
                    count += 1

        logging.info(f"Results from {count} samples have been aggregated.")

        # Need to get the mean contribution from the sum of all neuron's contribution that have been stored
        for key, value in storage.items():
            for token_type, aggregated_value in value.items():
                storage[key][token_type] = aggregated_value / count

        with open(Path(f"{self.destination_path}.pkl"), 'wb') as f:
            pickle.dump(storage, f)
