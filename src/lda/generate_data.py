from collections import defaultdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from typing import Annotated, Callable, Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import TextItem, IDItem
from tqdm import tqdm
import os
from experimaestro import Task, Param, Config, Meta, Constant
from experimaestro.launcherfinder import find_launcher
from experimaestro.generators import pathgenerator
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.letor.samplers import PairwiseSampleDataset
from xpmir.rankers.standard import BM25
from xpmir.papers.helpers.samplers import ValidationSample

from sampling import DiversePassagesSamplerWithHardNegatives
from utils import generic_dataset, check_pid_and_qid, get_token_types_spans, get_interesting_modules
from extractors import OutputsExtractorWithResiduals
from lda.config import LDAStudy
import gzip
import pickle
from functools import partial
import xpmir.interfaces.anserini as anserini

import logging

logging.basicConfig(level=logging.INFO)

def update_target_storage(data_storage, total_weights_storage, activations, spans, query_length, proba_diff):
    for layer_name, hidden_states in activations.items():
        # Parse layer and module names
        layer_nb = int(layer_name.split(".")[3])
        module_name = ".".join(layer_name.split(".")[4:])

        # Ensure defaultdict for layer/module
        layer_key = f"layer_{layer_nb}"
        module_storage = data_storage[layer_key][module_name] = data_storage[layer_key].get(module_name, defaultdict(dict))

        # Slice tokens
        cls_token = hidden_states[:, spans[0], :]
        query_tokens = hidden_states[:, spans[1], :]
        sep_1_token = hidden_states[:, spans[2], :]
        document_tokens = hidden_states[:, spans[3], :]
        sep_2_token = hidden_states[:, spans[4], :]

        # Common token updates
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        def update_token(name, tensor, mean=False, axis=1):
            if mean:
                tensor = torch.mean(tensor, dim=axis)
            if name not in module_storage:
                module_storage[name] = {
                    "streamed_numerator": 0,
                    "streamed_mean": 0
                }
            # Stream the numerator and the mean
            numpy_tensor = to_numpy(tensor)
            module_storage[name]["streamed_numerator"] += proba_diff / total_weights_storage["global"] * (numpy_tensor * numpy_tensor.T - module_storage[name]["streamed_numerator"])
            module_storage[name]["streamed_mean"] += proba_diff / total_weights_storage["global"] * (numpy_tensor - module_storage[name]["streamed_mean"])

        update_token("cls", cls_token.squeeze(dim=0))
        update_token("sep_1", sep_1_token.squeeze(dim=0)) 
        update_token("sep_2", sep_2_token.squeeze(dim=0)) 
        update_token("document", document_tokens, mean=True, axis=1) 

        # Handle query tokens
        for token in range(query_length):
            name = f"query_{token}"
            token_vector = query_tokens[:, token, :]
            if name not in module_storage:
                module_storage[name] = {
                    "streamed_numerator": 0,
                    "streamed_mean": 0
                }
            # Stream the numerator and the mean / Special treatment for the query tokens, they are streamed position-wise
            numpy_tensor = to_numpy(token_vector)
            module_storage[name]["streamed_numerator"] += proba_diff / total_weights_storage[name] * (numpy_tensor * numpy_tensor.T - module_storage[name]["streamed_numerator"])
            module_storage[name]["streamed_mean"] += proba_diff / total_weights_storage[name] * (numpy_tensor - module_storage[name]["streamed_mean"])

class GenerateMatrices(Task):    
    __xpmid__="src.evd.generate_data.generatematrices"

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    max_query_len: Param[int] = 3

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]

    device: Meta[Device] = DEFAULT_DEVICE

    positive_storage_path: Annotated[Path, pathgenerator("positives")]

    negative_storage_path: Annotated[Path, pathgenerator("strong_negatives")]

    falses_storage_path: Annotated[Path, pathgenerator("false_negatives_and_positives")]    

    version: Constant[int] = 2
    
    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model.eval()
        num_labels = model.config.num_labels
        # We determine in advance what activation function we'll need to parse the prediction as well as which label interests us
        if num_labels == 1:
            activation_fct = lambda x, dim: F.sigmoid(x)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            target_position = 1

        layer_names = get_interesting_modules(
            model=model,
        )

        extractor = OutputsExtractorWithResiduals(
            model=model,
            layer_names=layer_names
        )

        model.to(device_information.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        documents = prepare_dataset(self.dataset_name)
        topics = prepare_dataset(self.dataset_name + '.queries')

        positive_data_storage = defaultdict(list)
        negative_data_storage = defaultdict(list)
        false_negative_data_storage = defaultdict(list)
        false_positive_data_storage = defaultdict(list)

        # Global storage now uses defaultdict(dict) for each layer
        positive_global_storage = defaultdict(dict)
        negative_global_storage = defaultdict(dict)

        # Denominator dictionaries with pre-filled "global" and "query_{i}"
        positive_denominator = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})
        negative_denominator = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})

        positive_count = 0
        negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        # Initialize the Kahan compensations for each input part
        positive_kahan_compensations = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})
        negative_kahan_compensations = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})

        logging.info(f"Starting to gather the activations for the dataset {self.dataset_name}.")
        for qrel in tqdm(self.parsed_dataset.iter()):
            query_id = qrel.topics[0]
            query = topics.topic_ext(query_id)
            
            negative_data_storage[query_id] = list()
            positive_data_storage[query_id] = list()
            false_negative_data_storage[query_id] = list()
            false_positive_data_storage[query_id] = list()

            for passage_id in qrel.positives:
                passage = documents.documents.document_ext(passage_id)

                inputs = tokenizer(
                    query[TextItem].text,
                    passage[TextItem].text,
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(model.device)
                inputs.requires_grad = False

                outputs = extractor(inputs)
                proba_of_relevance = activation_fct(outputs.logits, dim=1)[0][target_position].detach().cpu().numpy()

                if proba_of_relevance > 0.5:
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    if spans[1].stop - spans[1].start > self.max_query_len:
                        spans[1] = slice(spans[1].start, spans[1].start + self.max_query_len)
                    query_length = spans[1].stop - spans[1].start

                    positive_data_storage[query[IDItem].id].append(passage[IDItem].id)

                    # Adjusts the sum of the weights using Kahan summation
                    proba_diff = abs(proba_of_relevance - 0.5)
                    compensated_sample = proba_diff - positive_kahan_compensations["global"]
                    compensated_sum = positive_denominator["global"] + compensated_sample
                    positive_kahan_compensations["global"] = (compensated_sum - positive_denominator["global"]) - compensated_sample
                    positive_denominator["global"] = compensated_sum

                    for i in range(query_length):
                        compensated_sample = proba_diff - positive_kahan_compensations[f"query_{i}"]
                        compensated_sum = positive_denominator[f"query_{i}"] + compensated_sample
                        positive_kahan_compensations[f"query_{i}"] = (compensated_sum - positive_denominator[f"query_{i}"]) - compensated_sample
                        positive_denominator[f"query_{i}"] = compensated_sum

                    update_target_storage(positive_global_storage, positive_denominator, extractor.outputs_store, spans, query_length, proba_diff)
                    positive_count += 1
                else:
                    # False negative
                    false_negative_data_storage[query[IDItem].id].append(passage[IDItem].id)
                    false_negative_count += 1
                    
            # Iterate through the negative passages
            for passage_id in qrel.negatives:
                passage = documents.documents.document_ext(passage_id)
                inputs = tokenizer(
                    query[TextItem].text, 
                    passage[TextItem].text,
                    max_length=128, # True max length is 512 but it gets too long to compute
                    truncation=True,
                    padding="max_length", # To get every sample to the same size
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(model.device)
                inputs.requires_grad = False
                outputs = extractor(inputs)
                proba_of_relevance = activation_fct(outputs.logits, dim=1)[0][target_position].detach().cpu().numpy()

                if proba_of_relevance < 0.5:
                    # If model agrees with annotations, we can pursue the process
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    if spans[1].stop - spans[1].start > self.max_query_len:
                        spans[1] = slice(spans[1].start, spans[1].start + self.max_query_len)
                    query_length = spans[1].stop - spans[1].start
                
                    # Accumulates the denominator for the EVD (shared by all modules / layers)
                    negative_data_storage[query[IDItem].id].append(passage[IDItem].id)
                    
                    # Adjusts the sum of the weights using Kahan summation
                    proba_diff = abs(proba_of_relevance - 0.5)
                    compensated_sample = proba_diff - negative_kahan_compensations["global"]
                    compensated_sum = negative_denominator["global"] + compensated_sample
                    negative_kahan_compensations["global"] = (compensated_sum - negative_denominator["global"]) - compensated_sample
                    negative_denominator["global"] = compensated_sum

                    for i in range(query_length):
                        compensated_sample = proba_diff - negative_kahan_compensations[f"query_{i}"]
                        compensated_sum = negative_denominator[f"query_{i}"] + compensated_sample
                        negative_kahan_compensations[f"query_{i}"] = (compensated_sum - negative_denominator[f"query_{i}"]) - compensated_sample
                        negative_denominator[f"query_{i}"] = compensated_sum

                    update_target_storage(negative_global_storage, negative_denominator, extractor.outputs_store, spans, query_length, proba_diff)
                    negative_count += 1
                else:
                    # False positive
                    false_positive_data_storage[query[IDItem].id].append(passage[IDItem].id)
                    false_positive_count += 1                            

        logging.info(f"Saving.")

        # Save global_storage to a json file
        if not self.positive_storage_path.exists():
            os.mkdir(self.positive_storage_path)
        with gzip.open(f'{self.positive_storage_path}/denominator.pkl.gz', 'wb') as f:
            pickle.dump(positive_denominator, f)
        with gzip.open(f'{self.positive_storage_path}/global_storage.pkl.gz', 'wb') as f:
            pickle.dump(positive_global_storage, f)
        with open(f'{self.positive_storage_path}/query-passage_pairs_storage.json', 'w') as f:
            json.dump(positive_data_storage, f)

        if not self.negative_storage_path.exists():
            os.mkdir(self.negative_storage_path)
        with gzip.open(f'{self.negative_storage_path}/denominator.pkl.gz', 'wb') as f:
            pickle.dump(negative_denominator, f)
        with gzip.open(f'{self.negative_storage_path}/global_storage.pkl.gz', 'wb') as f:
            pickle.dump(negative_global_storage, f)
        with open(f'{self.negative_storage_path}/query-passage_pairs_storage.json', 'w') as f:
            json.dump(negative_data_storage, f)

        if not self.falses_storage_path.exists():
            os.mkdir(self.falses_storage_path)
        with open(f'{self.falses_storage_path}/query-passage_false_positive_pairs_storage.json', 'w') as f:
            json.dump(false_positive_data_storage, f)
        with open(f'{self.falses_storage_path}/query-passage_false_negative_pairs_storage.json', 'w') as f:
            json.dump(false_negative_data_storage, f)

        logging.info(f"Number of pairs whose activations are stored: {positive_count + negative_count}; {positive_count} positives and {negative_count} negatives.")
        logging.info(f"Number of false positives (not stored): {false_positive_count}.")
        logging.info(f"Number of false negatives (not stored): {false_negative_count}.")


class GenerateMatricesWeakNegatives(Task):
    __xpmid__="src.evd.generate_data.generatematricesweaknegatives"

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]
    # We still need to access the filtered set of queries to grab weak negatives ONLY for these

    max_query_len: Param[int] = 3
    
    target_count_per_query: Param[int] = 15

    device: Meta[Device] = DEFAULT_DEVICE

    storage_path: Annotated[Path, pathgenerator("weak_negatives")]

    version: Constant[int] = 2

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model.eval()
        num_labels = model.config.num_labels
        # We determine in advance what activation function we'll need to parse the prediction as well as which label interests us
        if num_labels == 1:
            activation_fct = lambda x, dim: F.sigmoid(x)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            target_position = 1

        layer_names = get_interesting_modules(
            model=model,
        )

        extractor = OutputsExtractorWithResiduals(
            model=model,
            layer_names=layer_names
        )

        model.to(device_information.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        global_storage = defaultdict(dict)
        # Stores query passage pairs for later usage
        data_storage = defaultdict(list)

        denominator = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})
        kahan_compensations = defaultdict(float, {"global": 0.0, **{f"query_{i}": 0.0 for i in range(self.max_query_len)}})

        qrels = prepare_dataset(self.dataset_name + '.qrels')
        passages = prepare_dataset(self.dataset_name)
        topics = prepare_dataset(self.dataset_name + '.queries')

        logging.info(f"Starting to gather the activations for the dataset {self.dataset_name}.")
        nb_of_passages = passages.documents.documentcount
        total_count = 0
        for qrel in tqdm(self.parsed_dataset.iter()): # Only iterate through the filtered set of queries
            query_id = qrel.topics[0]
            query = topics.topic_ext(query_id)

            count = 0
            data_storage[query[IDItem].id] = []
            while count < self.target_count_per_query:
                # We continue until we break the loop because we have reached the target count of negatives per query
                passage = passages.documents.document_int(random.randint(0, nb_of_passages))
                
                while check_pid_and_qid(passage[IDItem].id, query[IDItem].id, qrels) or passage[TextItem].text == "":
                    passage = passages.documents.document_int(random.randint(0, nb_of_passages))

                inputs = tokenizer(
                    query[TextItem].text, 
                    passage[TextItem].text,
                    max_length=128, # True max length is 512 but it gets too long to compute
                    truncation=True,
                    padding="max_length", # To get every sample to the same size
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(model.device)
                inputs.requires_grad = False
                outputs = extractor(inputs)
                proba_of_relevance = activation_fct(outputs.logits, dim=1)[0][target_position].detach().cpu().numpy()
                if proba_of_relevance < 0.5: # Further uses the model to check that this is indeed a negative passage
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    if spans[1].stop - spans[1].start > self.max_query_len:
                        spans[1] = slice(spans[1].start, spans[1].start + self.max_query_len)
                    query_length = spans[1].stop - spans[1].start

                    count += 1
                    total_count += 1
                    data_storage[query[IDItem].id].append(passage[IDItem].id)

                    # Adjusts the sum of the weights using Kahan summation
                    proba_diff = abs(proba_of_relevance - 0.5)
                    compensated_sample = proba_diff - kahan_compensations["global"]
                    compensated_sum = denominator["global"] + compensated_sample
                    kahan_compensations["global"] = (compensated_sum - denominator["global"]) - compensated_sample
                    denominator["global"] = compensated_sum

                    for i in range(query_length):
                        compensated_sample = proba_diff - kahan_compensations[f"query_{i}"]
                        compensated_sum = denominator[f"query_{i}"] + compensated_sample
                        kahan_compensations[f"query_{i}"] = (compensated_sum - denominator[f"query_{i}"]) - compensated_sample
                        denominator[f"query_{i}"] = compensated_sum

                    update_target_storage(global_storage, denominator, extractor.outputs_store, spans, query_length, proba_diff)     

        logging.info(f"Saving.")

        # Save global_storage to a json file
        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        with gzip.open(f'{self.storage_path}/denominator.pkl.gz', 'wb') as f:
            pickle.dump(denominator, f)
        with gzip.open(f'{self.storage_path}/global_storage.pkl.gz', 'wb') as f:
            pickle.dump(global_storage, f)

        with open(f'{self.storage_path}/query-passage_pairs_storage.json', 'w') as f:
            json.dump(data_storage, f)

        logging.info(f"Number of pairs whose activations are stored: {total_count} weak negatives.")

class AgregateMatricesOutput(Config):
    __xpmid__="src.evd.generate_data.agregatematricesoutput"
    
    task: Meta[Config]

    per_dataset_storage_paths: Meta[Dict]
        
class AgregateMatrices(Task):    
    __xpmid__="src.evd.generate_data.agregatematrices"

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    outputs_generation: Param[List[Config]]

    positive_storage_path: Annotated[Path, pathgenerator("positive")]

    negative_storage_path: Annotated[Path, pathgenerator("negative")]

    version: Constant[int] = 2

    def task_outputs(self, dep: Callable[[Config], None]) -> AgregateMatricesOutput:
        per_dataset_storage_paths = dict()
        for output_generation in self.outputs_generation:
            if not output_generation.dataset_name in per_dataset_storage_paths.keys():
                per_dataset_storage_paths[output_generation.dataset_name] = dict()
            if output_generation.__class__.__name__ == "GenerateMatrices.XPMConfig":
                # Useful for the PCA, at the time of the projections we need to fetch the data
                per_dataset_storage_paths[output_generation.dataset_name]["positives"] = output_generation.positive_storage_path
                per_dataset_storage_paths[output_generation.dataset_name]["strong_negatives"] = output_generation.negative_storage_path
            elif output_generation.__class__.__name__ == "GenerateMatricesWeakNegatives.XPMConfig":
                per_dataset_storage_paths[output_generation.dataset_name]["weak_negatives"] = output_generation.storage_path
            else:
                raise ValueError(f"Not recognized `GenerateMatrices` subtask: {output_generation.__class__.__name__}")
            
        return dep(AgregateMatricesOutput.C(task=self, per_dataset_storage_paths=per_dataset_storage_paths))

    def execute(self):
        """Stream the averages of x*x^T (for the covariance) and x (for the mean) across the datasets, per levels of relevance / module / input part.
        Each dataset is given the same weight here, contrary to the previous step where each input was weighted by the strength of its probability of relevance.

        :raises ValueError: _description_
        """
        negative_storage_across_datasets = dict()
        positive_storage_across_datasets = dict()
        positive_sum_denominator = dict()
        negative_sum_denominator = dict()

        weak_negatives_storage_across_datasets = dict()
        weak_negatives_sum_denominator = dict()
        has_weak_negatives = False

        count = 0
        weak_negatives_count = 0
        logging.info("Start agregating the details from the list of datasets whose corresponding activations have been collected.")
        for output_generation in self.outputs_generation:
            if output_generation.__class__.__name__ == "GenerateMatrices.XPMValue":
                count += 1
                with gzip.open(f"{output_generation.positive_storage_path}/denominator.pkl.gz", 'rb') as f:
                    tmp_positive_denominator = pickle.load(f)
                for key, value in tmp_positive_denominator.items():
                    if key not in positive_sum_denominator.keys():
                        positive_sum_denominator[key] = 0
                    tmp = positive_sum_denominator[key]
                    positive_sum_denominator[key] += (value - tmp) / count

                with gzip.open(f"{output_generation.positive_storage_path}/global_storage.pkl.gz", 'rb') as f:
                    positive_data = pickle.load(f)
                for layer_nb, modules_dict in positive_data.items():
                    if layer_nb not in positive_storage_across_datasets.keys():
                        positive_storage_across_datasets[layer_nb] = dict()
                    for module_name, token_types_dict in modules_dict.items():
                        if module_name not in positive_storage_across_datasets[layer_nb].keys():
                            positive_storage_across_datasets[layer_nb][module_name] = dict()
                        for token_type, matrices_dict in token_types_dict.items():
                            if token_type not in positive_storage_across_datasets[layer_nb][module_name].keys():
                                # Because of the streaming, we need to sum the weighted numerators and the weighted means WITH their respective weight
                                positive_storage_across_datasets[layer_nb][module_name][token_type] = {
                                    "streamed_numerator": matrices_dict["streamed_numerator"].astype(np.float64),
                                    "streamed_mean": matrices_dict["streamed_mean"].astype(np.float64)
                                }
                            else:
                                tmp_numerator = positive_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"].astype(np.float64)
                                tmp_mean = positive_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"].astype(np.float64)

                                positive_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"] += (matrices_dict["streamed_numerator"].astype(np.float64) - tmp_numerator) / count
                                positive_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"] += (matrices_dict["streamed_mean"].astype(np.float64) - tmp_mean) / count

                with gzip.open(f"{output_generation.negative_storage_path}/denominator.pkl.gz", 'rb') as f:
                    tmp_negative_denominator = pickle.load(f)
                for key, value in tmp_negative_denominator.items():
                    if key not in negative_sum_denominator.keys():
                        negative_sum_denominator[key] = 0
                    tmp = negative_sum_denominator[key]
                    negative_sum_denominator[key] += (value - tmp) / count
                with gzip.open(f"{output_generation.negative_storage_path}/global_storage.pkl.gz", 'rb') as f:
                    negative_data = pickle.load(f)
                for layer_nb, modules_dict in negative_data.items():
                    if layer_nb not in negative_storage_across_datasets.keys():
                        negative_storage_across_datasets[layer_nb] = dict()
                    for module_name, token_types_dict in modules_dict.items():
                        if module_name not in negative_storage_across_datasets[layer_nb].keys():
                            negative_storage_across_datasets[layer_nb][module_name] = dict()
                        for token_type, matrices_dict in token_types_dict.items():
                            if token_type not in negative_storage_across_datasets[layer_nb][module_name].keys():
                                negative_storage_across_datasets[layer_nb][module_name][token_type] = {
                                    "streamed_numerator": matrices_dict["streamed_numerator"].astype(np.float64),
                                    "streamed_mean": matrices_dict["streamed_mean"].astype(np.float64) 
                                }
                            else:
                                tmp_numerator = negative_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"].astype(np.float64)
                                tmp_mean = negative_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"].astype(np.float64)
                                # Weights are all equal here, so we can just do a simple
                                negative_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"] += (matrices_dict["streamed_numerator"].astype(np.float64) - tmp_numerator) / count
                                negative_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"] += (matrices_dict["streamed_mean"].astype(np.float64) - tmp_mean) / count

            elif output_generation.__class__.__name__ == "GenerateMatricesWeakNegatives.XPMValue":
                weak_negatives_count += 1
                has_weak_negatives = True
                with gzip.open(f"{output_generation.storage_path}/denominator.pkl.gz", 'rb') as f:
                    tmp_denominator = pickle.load(f)
                for key, value in tmp_denominator.items():
                    if key not in weak_negatives_sum_denominator.keys():
                        weak_negatives_sum_denominator[key] = 0
                    tmp = weak_negatives_sum_denominator[key]
                    weak_negatives_sum_denominator[key] += (value - tmp) / weak_negatives_count

                with gzip.open(f"{output_generation.storage_path}/global_storage.pkl.gz", 'rb') as f:
                    data = pickle.load(f)
                for layer_nb, modules_dict in data.items():
                    if layer_nb not in weak_negatives_storage_across_datasets.keys():
                        weak_negatives_storage_across_datasets[layer_nb] = dict()
                    for module_name, token_types_dict in modules_dict.items():
                        if module_name not in weak_negatives_storage_across_datasets[layer_nb].keys():
                            weak_negatives_storage_across_datasets[layer_nb][module_name] = dict()
                        for token_type, matrices_dict in token_types_dict.items():
                            if token_type not in weak_negatives_storage_across_datasets[layer_nb][module_name].keys():
                                weak_negatives_storage_across_datasets[layer_nb][module_name][token_type] = {
                                    "streamed_numerator": matrices_dict["streamed_numerator"].astype(np.float64),
                                    "streamed_mean": matrices_dict["streamed_mean"].astype(np.float64) 
                                }
                            else:
                                tmp_numerator = weak_negatives_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"].astype(np.float64)
                                tmp_mean = weak_negatives_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"].astype(np.float64)
                                # Weights are all equal here,
                                weak_negatives_storage_across_datasets[layer_nb][module_name][token_type]["streamed_numerator"] += (matrices_dict["streamed_numerator"].astype(np.float64) - tmp_numerator) / weak_negatives_count
                                weak_negatives_storage_across_datasets[layer_nb][module_name][token_type]["streamed_mean"] += (matrices_dict["streamed_mean"].astype(np.float64) - tmp_mean) / weak_negatives_count
            else:
                raise ValueError("The output_generation class is not recognized.")
        
        logging.info("Saving the agregations.")
        # Saving the averaged statistics per relevance levels across input parts / modules / layers
        if not Path(self.positive_storage_path).exists():
            os.mkdir(self.positive_storage_path)
        with gzip.open(Path(f'{self.positive_storage_path}/denominator_sum.pkl.gz'), 'wb') as f:
            pickle.dump(positive_sum_denominator, f)
        with gzip.open(Path(f'{self.positive_storage_path}/global_storage.pkl.gz'), 'wb') as f:
            pickle.dump(positive_storage_across_datasets, f)

        if not Path(self.negative_storage_path).exists():
            os.mkdir(self.negative_storage_path)
        with gzip.open(Path(f'{self.negative_storage_path}/denominator_sum.pkl.gz'), 'wb') as f:
            pickle.dump(negative_sum_denominator, f)
        with gzip.open(Path(f'{self.negative_storage_path}/global_storage.pkl.gz'), 'wb') as f:
            pickle.dump(negative_storage_across_datasets, f)

        if has_weak_negatives:
            weak_negatives_storage_path = self.negative_storage_path.parent / "weak_negative"
            logging.info("Saving the agregations for the weak negatives too.")
            if not Path(weak_negatives_storage_path).exists():
                os.mkdir(weak_negatives_storage_path)
            with gzip.open(Path(f'{weak_negatives_storage_path}/denominator_sum.pkl.gz'), 'wb') as f:
                pickle.dump(weak_negatives_sum_denominator, f)
            with gzip.open(Path(f'{weak_negatives_storage_path}/global_storage.pkl.gz'), 'wb') as f:
                pickle.dump(weak_negatives_storage_across_datasets, f)

def gather_matrices(cfg: LDAStudy):
    """For every qrels in each of the datasets, gather the outputs from the attention and the MLP blocks (for each layer of the model)
    and aggregate them. The aggregation is done by summing the outer product of the activations, weighted by the final probability of relevance of the qrels,
    of the tokens of the same type.
    For CLS, SEP1, SEP2: Simple summation
    For Query's tokens: Simple summation but position-wise (over Q1, over Q2, etc. until Q_(max_query_len) )
    For Document's tokens: Simple summation of the average activation across all the document's tokens 
    """
    launcher_sampling =  find_launcher(cfg.gather_matrices.sampling_requirements)
    launcher_generation = find_launcher(cfg.gather_matrices.generation_requirements)
    launcher_indexation = find_launcher(cfg.gather_matrices.indexation_requirements)
    generation_outputs = list()

    # Define BM25 retriever
    base_model = BM25.C()
    index_builder = anserini.index_builder(launcher=launcher_indexation)

    retriever = partial(
        anserini.retriever,
        index_builder,
        model=base_model,
    )  #: Anserini based retrievers

    if cfg.gather_matrices.check_datasets_list():
        for dataset_config in cfg.gather_matrices.list_of_datasets:
            sample_config = ValidationSample(size=cfg.gather_matrices.max_queries) 
            datamaestro_dataset = generic_dataset(sample_config, dataset_config["dataset_name"]) # Randomly sample queries from the dataset to limit its size
            preprocessed_dataset = DiversePassagesSamplerWithHardNegatives.C(
                dataset=datamaestro_dataset,
                retriever=retriever(datamaestro_dataset.documents, k=200), # Hopefully that's enough to cover the additional negatives needed
                relevance_levels=dataset_config["relevance_levels"],
                max_passages_per_relevance_level=dataset_config["max_passages_per_relevance_level"],
                max_passages_per_query=dataset_config["max_passages_per_query"],
            ).submit(launcher=launcher_sampling)

            generation_output = GenerateMatrices.C(
                ranker_id=cfg.ranker_id,
                base_hf_id=cfg.base_hf_id,
                dataset_name=dataset_config["dataset_name"],
                max_query_len=cfg.gather_matrices.max_query_len,
                parsed_dataset=preprocessed_dataset,    
                device=cfg.device,
            ).submit(launcher=launcher_generation)
            generation_outputs.append(generation_output)

            if cfg.gather_matrices.get_weak_negatives:
                generation_outputs.append(
                    GenerateMatricesWeakNegatives.C(
                        ranker_id=cfg.ranker_id,
                        base_hf_id=cfg.base_hf_id,
                        dataset_name=dataset_config["dataset_name"],
                        parsed_dataset=preprocessed_dataset,
                        max_query_len=cfg.gather_matrices.max_query_len,
                        target_count_per_query=dataset_config["max_passages_per_relevance_level"],
                        device=cfg.device,
                    ).submit(launcher=launcher_generation)
                )

    launcher_agregation = find_launcher(cfg.post_processing.requirements)
    agregate_output = AgregateMatrices.C(
        ranker_id=cfg.ranker_id,
        base_hf_id=cfg.base_hf_id,
        outputs_generation=generation_outputs,
    ).submit(launcher=launcher_agregation)

    return agregate_output