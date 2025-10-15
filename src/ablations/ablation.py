from .utils import AblationOutput, PrunedModelForCrossScorer, get_relevance_levels
from src.utils import get_token_types_spans, INPUT_PART_TO_POSITION

import os
import json
from typing import Callable, Dict, List, Annotated
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import TextItem, PairwiseSampleDataset
from tqdm import tqdm
from experimaestro import Task, Param, Config, Meta
from experimaestro.generators import pathgenerator
from pathlib import Path


import logging

logging.basicConfig(level=logging.INFO)

class AblationAttention(Task):
    cut_attention_from: Param[List[str]]

    cut_attention_to: Param[List[str]]

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]

    max_seq_length: Param[int]
    
    start_layer: Param[int]

    end_layer: Param[int]

    device: Meta[Device] = DEFAULT_DEVICE

    output_path: Annotated[Path, pathgenerator("output")]
    
    def task_outputs(self, dep: Callable[[Config], None]) -> AblationOutput:
        return dep(AblationOutput.C(task=self.C()))

    def execute(self):
        for direction in self.cut_attention_from:
            assert direction in INPUT_PART_TO_POSITION.keys(), f"Position {direction} doesn't correspond to an input part."
        for direction in self.cut_attention_to:
            assert direction in INPUT_PART_TO_POSITION.keys(), f"Position {direction} doesn't correspond to an input part."

        # Map the parsed dataset and its passages to the qrels from the dataset
        parsed_dataset_with_qrels = get_relevance_levels(self.dataset_name, self.parsed_dataset)
        self.device.execute(self.device_execute, parsed_dataset_with_qrels)

    def device_execute(self, device_information: DeviceInformation, parsed_dataset_with_qrels: Dict[str, Dict[str, int]]):
        original_model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        ablation_model = PrunedModelForCrossScorer(
            model=model,
            start_layer=self.start_layer,
            end_layer=self.end_layer
        )
        original_model.eval()
        ablation_model.eval()

        if original_model.config.num_labels == 1:
            activation_fct = lambda x, dim: F.sigmoid(x)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            target_position = 1

        original_model.to(device_information.device)
        ablation_model.to(device_information.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        passages = prepare_dataset(self.dataset_name)
        topics = prepare_dataset(self.dataset_name + '.queries')

        logging.info(f"Starting the ablations for the dataset {self.dataset_name}.")
        proba_diff = 0
        num_samples = 0
        # Check for already processed (query_id, pid) pairs
        processed_pairs = set()
        log_file = f"{self.output_path}/ablation_logs.jsonl"
        # If logs exist, load them and accumulate processed pairs and proba_diff
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        processed_pairs.add((entry["query_id"], entry["passage_id"]))
                        proba_diff += abs(entry["original_proba"] - entry["ablation_proba"])
                        num_samples += 1
                logging.info(f"Loaded {num_samples} existing logs from {log_file}.")
            except Exception as e:
                logging.warning(f"Could not load existing logs from {log_file}: {e}")
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
       
        for query_id, pid_to_rel in tqdm(parsed_dataset_with_qrels.items()):
            query = topics.topic_ext(query_id)
            for pid, rel in tqdm(pid_to_rel.items()):
                if (query_id, pid) in processed_pairs:
                    continue  # Skip already processed pairs
                passage = passages.documents.document_ext(pid)
                inputs = tokenizer(
                    query[TextItem].text, 
                    passage[TextItem].text,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(device_information.device)
                inputs.requires_grad = False
                spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                positions_to_cut_attention_from = [spans[INPUT_PART_TO_POSITION[direction]] for direction in self.cut_attention_from]
                positions_to_cut_attention_to = [spans[INPUT_PART_TO_POSITION[direction]] for direction in self.cut_attention_to] if self.cut_attention_to else []
                attention_mask = torch.ones((self.max_seq_length, self.max_seq_length)).to(device_information.device)
                for slice_from in positions_to_cut_attention_from:
                    if positions_to_cut_attention_to:
                        for slice_to in positions_to_cut_attention_to:
                            attention_mask[slice_from, slice_to] = 0
                    else:
                        attention_mask[slice_from, :] = 0

                original_logits = original_model(**inputs).logits.detach().cpu()
                original_proba = activation_fct(original_logits, dim=1)[0][target_position].detach().cpu().numpy()

                inputs["attention_scores_mask"] = attention_mask
                ablated_logits = ablation_model(**inputs).logits.detach().cpu()
                ablated_proba = activation_fct(ablated_logits, dim=1)[0][target_position].detach().cpu().numpy()
                log_entry = {
                    "query_id": query_id,
                    "passage_id": pid,
                    "passage_relevance": rel,
                    "original_logits": original_logits.numpy().tolist(),
                    "original_proba": float(original_proba),
                    "ablation_logits": ablated_logits.numpy().tolist(),
                    "ablation_proba": float(ablated_proba)
                }
                proba_diff += abs(original_proba - ablated_proba)
                num_samples += 1

                # Save log after each entry for robustness (append as JSONL)
                with open(f"{self.output_path}/ablation_logs.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
        
        logging.info(f"Average difference in probabilities: {proba_diff/num_samples}")

        if not self.output_path.exists():
            os.mkdir(self.output_path)
        value_file = f"{self.output_path}/averaged_diff.npy"
        np.save(value_file, proba_diff/num_samples)
        logging.info(f"Logs saved to {log_file} and averaged difference value saved to {value_file}.")