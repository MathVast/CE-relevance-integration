import gc
import gzip
import os
from pathlib import Path
import pickle
import random
from typing import Annotated, Callable
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import TextItem, IDItem
from tqdm import tqdm
from experimaestro import Task, Param, Meta, Constant
from experimaestro.generators import pathgenerator
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.letor.samplers import PairwiseSampleDataset
from utils import check_pid_and_qid, get_token_types_spans, untuple
from attention.utils import load_written_ids, write_data_line

import logging

logging.basicConfig(level=logging.INFO)

class GrabAttentionPatterns(torch.nn.Module):
    # Hook used to mask the output of some layer given a pruning scheme
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.attention_probs_storage = dict()
        self.hooks_handles = list()

        for layer_nb in range(len(self.model.bert.encoder.layer)):
            module_name = f"bert.encoder.layer.{layer_nb}.attention.self.dropout"
            layer = dict([*self.model.named_modules()])[module_name]
            self.hooks_handles.append(layer.register_forward_hook(self.register_attention_probs(module_name)))

    def register_attention_probs(self, module_name) -> Callable:
        def hook(model, input, output):
            input = untuple(input)
            self.attention_probs_storage[module_name] = input
        return hook
    
    @property
    def items(self):
        return self.attention_probs_storage
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()
    
    def clear_items(self):
        del self.attention_probs_storage
        gc.collect()
        torch.cuda.empty_cache()
        self.attention_probs_storage = dict()
    
    def forward(self, inputs):
        model_outputs = self.model(**inputs)
        return model_outputs


class GenerateAttentionPatterns(Task):  
    __xpmid__="src.evd.attention_patterns_generation.generateattentionpatterns"

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]

    device: Meta[Device] = DEFAULT_DEVICE

    attentions_storage_path: Annotated[Path, pathgenerator("attentions")]

    data_storage_path: Annotated[Path, pathgenerator("data")]

    version: Constant[int] = 2
    
    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model.eval()
        num_labels = model.config.num_labels
        # Determine activation function and label extraction
        if num_labels == 1:
            activation_fct = lambda x, dim: torch.sigmoid(x)
            get_pred_label = lambda x: int(x > 0.5)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            get_pred_label = lambda x: int(torch.argmax(x, dim=1))
            target_position = 1


        grab_outputs = GrabAttentionPatterns(model)
        model.to(device_information.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        documents = prepare_dataset(self.dataset_name)
        topics = prepare_dataset(self.dataset_name + '.queries')

        positive_count = 0
        negative_count = 0
        false_positive_count = 0
        false_negative_count = 0
        query_count = 0

        # Make sure output directory exists
        os.makedirs(self.data_storage_path, exist_ok=True)
        os.makedirs(self.attentions_storage_path, exist_ok=True)

        # Open data files for relevant and non-relevant
        relevant_data_path = os.path.join(self.data_storage_path, "relevant.jsonl")
        nonrelevant_data_path = os.path.join(self.data_storage_path, "non_relevant.jsonl")
        relevant_data_f = open(relevant_data_path, 'a', encoding='utf-8')
        nonrelevant_data_f = open(nonrelevant_data_path, 'a', encoding='utf-8')
        already_written_relevant = load_written_ids(relevant_data_path)
        already_written_nonrelevant = load_written_ids(nonrelevant_data_path)

        # We'll keep track of opened attention files
        attention_files = {
            "relevant": {},
            "non_relevant": {}
        }

        attention_keys_seen = set()  # to track which attention keys we've encountered

        logging.info(f"Starting to gather the activations for the dataset {self.parsed_dataset.id}.")

        for qrel in tqdm(self.parsed_dataset.iter()):
            query_id = qrel.topics[0] 
            query = topics.topic_ext(query_id)
            query_count += 1

            # Process positive passages
            for passage_id in qrel.positives:
                if (passage_id, query_id) in already_written_relevant:
                    continue

                passage = documents.documents.document_ext(passage_id)
                if passage[TextItem].text == "":
                    continue

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
                outputs = grab_outputs(inputs)
                pred_label = get_pred_label(activation_fct(outputs.logits, dim=1))

                if pred_label == 1:
                    positive_count += 1
                    relevance = "relevant"
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    write_data_line(relevant_data_f, pid=passage_id, qid=query_id, spans=spans)

                    # On first encounter, open attention files if not opened yet
                    if not attention_files[relevance]:
                        attention_keys = list(grab_outputs.attention_probs_storage.keys())
                        for key in attention_keys:
                            path = f"{self.attentions_storage_path}/{relevance}_{key}.pkl.gz"
                            attention_files[relevance][key] = gzip.open(path, "ab")
                        attention_keys_seen.update(attention_keys)

                    # Write each attention prob immediately with metadata
                    for key, value in grab_outputs.attention_probs_storage.items():
                        attn_dict = {
                            "pid": passage_id,
                            "qid": query_id,
                            "attention": value.detach().cpu().numpy(),
                        }
                        pickle.dump(attn_dict, attention_files[relevance][key])
                else:
                    false_negative_count += 1
                
                grab_outputs.clear_items()

            # Process negative passages
            for passage_id in qrel.negatives:
                if (passage_id, query_id) in already_written_nonrelevant:
                    continue
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
                outputs = grab_outputs(inputs)
                pred_label = get_pred_label(activation_fct(outputs.logits, dim=1))

                if pred_label == 0:
                    negative_count += 1
                    relevance = "non_relevant"
                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    write_data_line(nonrelevant_data_f, pid=passage_id, qid=query_id, spans=spans)

                    if not attention_files[relevance]:
                        attention_keys = list(grab_outputs.attention_probs_storage.keys())
                        for key in attention_keys:
                            path = f"{self.attentions_storage_path}/{relevance}_{key}.pkl.gz"
                            attention_files[relevance][key] = gzip.open(path, "ab")
                        attention_keys_seen.update(attention_keys)

                    # Write each attention prob immediately with metadata
                    for key, value in grab_outputs.attention_probs_storage.items():
                        attn_dict = {
                            "pid": passage_id,
                            "qid": query_id,
                            "attention": value.detach().cpu().numpy(),
                        }
                        pickle.dump(attn_dict, attention_files[relevance][key])
                else:
                    false_positive_count += 1
                grab_outputs.clear_items()

        logging.info(f"Finished parsing dataset {self.dataset_name}: {positive_count} positive samples and {negative_count} negative samples over {query_count} queries.")
        logging.info(f"Saving.")

        # Close all open files
        relevant_data_f.close()
        nonrelevant_data_f.close()
        for rel_key, files_dict in attention_files.items():
            for f in files_dict.values():
                f.close()

class GenerateAttentionPatternsWeakNegatives(Task):
    __xpmid__="src.evd.attention_patterns_generation.generateattentionpatternsweaknegatives"
    
    ranker_id: Param[str] 

    base_hf_id: Param[str]

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]
    
    target_count_per_query: Param[int] = 15

    device: Meta[Device] = DEFAULT_DEVICE

    attentions_storage_path: Annotated[Path, pathgenerator("attentions")]

    data_storage_path: Annotated[Path, pathgenerator("data")]

    version: Constant[int] = 2

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        os.makedirs(self.data_storage_path, exist_ok=True)
        os.makedirs(self.attentions_storage_path, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model.eval()
        num_labels = model.config.num_labels

        if num_labels == 1:
            activation_fct = lambda x, dim: F.sigmoid(x)
            get_pred_label = lambda x: int(x > 0.5)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            get_pred_label = lambda x: int(torch.argmax(x, dim=1))
            target_position = 1

        grab_outputs = GrabAttentionPatterns(model)
        model.to(device_information.device)

        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        qrels = prepare_dataset(self.dataset_name + '.qrels')
        passages = prepare_dataset(self.dataset_name)
        nb_of_passages = passages.documents.documentcount
        topics = prepare_dataset(self.dataset_name + '.queries')

        # Open output files for streaming
        data_path = os.path.join(self.data_storage_path, "weak_negative.jsonl")
        weak_negative_data_f = open(data_path, 'a', encoding='utf-8')
        already_written = load_written_ids(data_path)

        attention_files = {}

        negative_count = 0
        false_positive_count = 0
        query_count = 0

        for qrel in tqdm(self.parsed_dataset.iter()):
            query_id = qrel.topics[0]
            query = topics.topic_ext(query_id)

            query_count += 1
            for _ in range(self.target_count_per_query):
                passage = passages.documents.document_int(random.randint(0, nb_of_passages))
                while check_pid_and_qid(passage[IDItem].id, query[IDItem].id, qrels) or (passage[IDItem].id, query[IDItem].id) in already_written or passage[TextItem].text == "":
                    passage = passages.documents.document_int(random.randint(0, nb_of_passages))

                inputs = tokenizer(
                    query[TextItem].text,
                    passage[TextItem].text,
                    max_length=128,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(device_information.device)
                inputs.requires_grad = False
                outputs = grab_outputs(inputs)
                pred_label = get_pred_label(activation_fct(outputs.logits, dim=1))

                if pred_label == 1:
                    false_positive_count += 1
                    continue
                else:
                    relevance = "weak_negative"
                    negative_count += 1

                    spans = get_token_types_spans(inputs["input_ids"], tokenizer)
                    write_data_line(weak_negative_data_f, pid=passage[IDItem].id,  qid=query[IDItem].id, spans=spans)

                    if not attention_files:
                        attention_keys = list(grab_outputs.attention_probs_storage.keys())
                        for key in attention_keys:
                            path = f"{self.attentions_storage_path}/{relevance}_{key}.pkl.gz"
                            attention_files[key] = gzip.open(path, "ab")

                    # Write each attention prob immediately with metadata
                    for key, value in grab_outputs.attention_probs_storage.items():
                        attn_dict = {
                            "pid": passage[IDItem].id,
                            "qid": query_id,
                            "attention": value.detach().cpu().numpy(),
                        }
                        pickle.dump(attn_dict, attention_files[key])

                grab_outputs.clear_items()

        logging.info(f"Finished parsing dataset {self.dataset_name}: {negative_count} negative samples over {query_count} queries.")
        logging.info(f"Saving.")

        # Close files
        weak_negative_data_f.close()
        for f in attention_files.values():
            f.close()
