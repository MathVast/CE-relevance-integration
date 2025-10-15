import os
from pathlib import Path
from typing import Annotated, List
import zlib
from experimaestro import Task, Param, Config, Constant 
from experimaestro.generators import pathgenerator
import numpy as np
import pandas as pd
from scipy.stats import entropy
from transformers import AutoTokenizer
from datamaestro_text.data.ir import TextItem
from datamaestro import prepare_dataset
import json
import re
import pickle
import gzip

import logging

logging.basicConfig(level=logging.INFO)

class SummarizeAttentionPatterns(Task):
    generation_outputs: Param[List[Config]]

    base_hf_id: Param[str]

    start_layer: Param[int]

    end_layer: Param[int]

    relevance_key_to_level: Constant[List[List]] = [["relevant", 1], ["non_relevant", 0], ["weak_negative", -1]]

    storage_path: Annotated[Path, pathgenerator("dataframes")]

    def execute(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        """Summarize the attention patterns and save them to disk."""
        meta = {}
        storage = {}
        # Loading the meta data, ie. query-passage pairs across the different storages
        for generation_output in self.generation_outputs:
            logging.info(f"Processing metadata for: {generation_output.dataset_name}")
            if generation_output.dataset_name not in meta:
                meta[generation_output.dataset_name] = {}
            for file in generation_output.data_storage_path.iterdir():
                if "non_relevant" in file.name:
                    key = "non_relevant"
                elif "weak" in file.name:
                    key = "weak_negative"
                else:
                    key = "relevant"
                with open(file, 'r') as f:
                    logging.info(f"Loading {file}")
                    meta[generation_output.dataset_name][key] = [json.loads(line) for line in f]

            if generation_output.dataset_name not in storage:
                storage[generation_output.dataset_name] = {}
            
            logging.info(f"Processing attention patterns from: {generation_output.attentions_storage_path}")
            for file in generation_output.attentions_storage_path.iterdir():
                if "non_relevant" in file.name:
                    key = "non_relevant"
                elif "weak" in file.name:
                    key = "weak_negative"
                else:
                    key = "relevant"
                if key not in storage[generation_output.dataset_name]:
                    storage[generation_output.dataset_name][key] = dict()
                if file.is_file():
                    match = re.search(r'layer\.(\d+)\.attention', file.name)
                    layer_num = int(match.group(1)) if match else None
                    logging.info(f"Found file: {file}")
                    storage[generation_output.dataset_name][key][layer_num] = file
        
        # At this point, storage contains the attention patterns for all the queries across all the datasets
        # and meta contains the metadata for all the queries across all the datasets

        logging.info("Processing attention patterns across all layers, heads and samples.")
        attention_data = list()
        entropy_data = list()
        duplicate_token_data = list()
        
        # Loop over layers
        for dataset_name, per_dataset_storage in storage.items():
            queries = prepare_dataset(dataset_name + ".queries")
            documents = prepare_dataset(dataset_name)

            for relevance_level, layer_to_files in per_dataset_storage.items():
                for layer, file in layer_to_files.items():
                    logging.info(f"Processing layer {layer} for relevance level {relevance_level} in dataset {dataset_name}")

                    # Load the attention patterns for this layer
                    idx = 0
                    skipped_samples = 0
                    with gzip.open(file, "rb") as fp:
                        while True:
                            try:
                                obj = pickle.load(fp)
                                # process obj here
                                
                                # Loop over heads
                                if meta[dataset_name][relevance_level][idx]["pid"] != obj["pid"] or meta[dataset_name][relevance_level][idx]["qid"] != obj["qid"]:
                                    logging.info(f"Mismatch in pid or qid for dataset {dataset_name}, key {relevance_level}, idx {idx}. Expected pid: {meta[dataset_name][relevance_level][idx]['pid']}, qid: {meta[dataset_name][relevance_level][idx]['qid']}, got pid: {obj['pid']}, qid: {obj['qid']}")
                                    skipped_samples += 1
                                    idx += 1
                                    continue
                                query_slice = slice(meta[dataset_name][relevance_level][idx]["spans"][1]["start"], meta[dataset_name][relevance_level][idx]["spans"][1]["end"])
                                doc_slice = slice(meta[dataset_name][relevance_level][idx]["spans"][3]["start"], meta[dataset_name][relevance_level][idx]["spans"][3]["end"])

                                if doc_slice.start == doc_slice.stop:
                                    skipped_samples += 1
                                    idx += 1
                                    logging.info(f"Skipping sample {idx} for dataset {dataset_name}, relevance level {relevance_level}, layer {layer} due to empty document.")
                                    continue

                                for head_nb, head in enumerate(obj["attention"][0]):    

                                    # Query pays attention to Document <=> Document send its information to Query
                                    # Create a slice from the dictionary meta[dataset_name][relevance_level][idx]["spans"][1] using "start" and "end" keys
                                    mean_query_to_doc_attention = np.max(head[query_slice, doc_slice].flatten(), axis=0)

                                    # Document pays attention to Query <=> Query send its information to Document
                                    mean_doc_to_query_attention = np.max(head[doc_slice, query_slice].flatten(), axis=0)
                                    qid = meta[dataset_name][relevance_level][idx]['qid']
                                    pid = meta[dataset_name][relevance_level][idx]['pid']
                                    
                                    attention_data.append([qid, pid, layer, head_nb, relevance_level, mean_query_to_doc_attention, mean_doc_to_query_attention])

                                    # 2: Get the the entropy of attention values between query and document (both ways):
                                    query_to_doc_probas = head[query_slice, doc_slice]
                                    doc_to_query_probas = head[doc_slice, query_slice]
                                    
                                    sum_query_to_doc_probas = np.sum(query_to_doc_probas, axis=1, keepdims=True)
                                    sum_doc_to_query_probas = np.sum(doc_to_query_probas, axis=1, keepdims=True)

                                    # Turn them into proba distributions for the entropy
                                    normalized_query_to_doc_probas = query_to_doc_probas / sum_query_to_doc_probas
                                    normalized_doc_to_query_probas = doc_to_query_probas / sum_doc_to_query_probas

                                    query = queries.topic_ext(qid)[TextItem].text
                                    passage = documents.documents.document_ext(pid)[TextItem].text
                                    input_ids = tokenizer(
                                        query, 
                                        passage,
                                        max_length=128, # True max length is 512 but it gets too long to compute
                                        truncation=True,
                                        padding=False, # To get every sample to the same size
                                        return_attention_mask=False,
                                    )["input_ids"]

                                    mean_query_to_doc_entropy = sum_query_to_doc_probas * entropy(normalized_query_to_doc_probas, axis=1)
                                    mean_doc_to_query_entropy = sum_doc_to_query_probas * entropy(normalized_doc_to_query_probas, axis=1)
                                    
                                    np.nan_to_num(mean_query_to_doc_entropy, nan=0., copy=False)
                                    np.nan_to_num(mean_doc_to_query_entropy, nan=0., copy=False)
                                    entropy_data.append([qid, pid, layer, head_nb, relevance_level, np.sum(mean_query_to_doc_entropy)/np.sum(sum_query_to_doc_probas), np.sum(mean_doc_to_query_entropy)/np.sum(sum_doc_to_query_probas)])

                                    # 3: Get the sum of attention between duplicate tokens across query and document (both ways):
                                    query_ids = input_ids[query_slice]
                                    passage_ids = input_ids[doc_slice]
                                    
                                    mask = np.expand_dims(query_ids, axis=1) == np.expand_dims(passage_ids, axis=0)
                                    filtered_normalized_query_to_doc_probas = np.nan_to_num(np.sum(normalized_query_to_doc_probas * mask, axis=1), nan=0., copy=True)
                                    filtered_normalized_doc_to_query_probas = np.nan_to_num(np.sum(normalized_doc_to_query_probas * mask.T, axis=1), nan=0., copy=True)

                                    duplicate_token_query = np.sum(sum_query_to_doc_probas * filtered_normalized_query_to_doc_probas) / np.sum(sum_query_to_doc_probas)
                                    duplicate_token_passage = np.sum(sum_doc_to_query_probas * filtered_normalized_doc_to_query_probas) / np.sum(sum_doc_to_query_probas)

                                    duplicate_token_data.append([qid, pid, layer, head_nb, relevance_level, duplicate_token_query, duplicate_token_passage])
                                idx += 1
                                if idx % 1000 == 0:
                                    logging.info(f"Processed {idx} samples for dataset {dataset_name}, relevance level {relevance_level}, layer {layer}")

                            except EOFError:
                                break
                            except (pickle.UnpicklingError, zlib.error, OSError) as e:
                                logging.error(f"Error loading pickle from {file}: {e}")
                                break
                    logging.info(f"Skipped {skipped_samples} samples for dataset {dataset_name}, relevance level {relevance_level}, layer {layer}")

            
        logging.info("Saving.")
        attention_data = pd.DataFrame(attention_data, columns=["qid", "pid", "layer", "head", "rel", "q2d", "d2q"])

        duplicate_token_data = pd.DataFrame(duplicate_token_data, columns=["qid", "pid", "layer", "head", "rel", "q2d", "d2q"])
        entropy_data = pd.DataFrame(entropy_data, columns=["qid", "pid", "layer", "head", "rel", "q2d", "d2q"])

        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        with gzip.open(f"{self.storage_path}/attention.pkl.gz", "wb") as fp:
            pickle.dump(attention_data, fp)

        with gzip.open(f"{self.storage_path}/attention_entropy.pkl.gz", "wb") as fp:
            pickle.dump(entropy_data, fp)

        with gzip.open(f"{self.storage_path}/attention_duplicate_token.pkl.gz", "wb") as fp:
            pickle.dump(duplicate_token_data, fp)