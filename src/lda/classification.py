import csv
import re
import pandas as pd
import pickle
import gzip
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)
from typing import Dict, List,  Annotated
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from experimaestro import Task, Param, Config, Meta, Constant
from experimaestro.generators import pathgenerator

from src.lda.generate_data import AgregateMatricesOutput
from src.utils import get_interesting_modules
from src.extractors import OutputsExtractorWithResiduals
from src.lda.utils import GetActivationsOutput, iterate_over_pairs_and_store_activations

import logging

logging.basicConfig(level=logging.INFO)

class ComputeClassification(Task):
    agregation_output: Param[AgregateMatricesOutput] # We need it in case of weak negatives as they have been sampled randomly

    list_of_datasets: Param[List[Dict]]

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    collect_direction: Param[Config]

    target_module: Param[str]

    input_parts: Param[List[str]]

    start_layer: Param[int]

    end_layer: Param[int]

    device: Meta[Device] = DEFAULT_DEVICE

    version: Constant[int] = 2

    storage_path: Annotated[Path, pathgenerator("plots")]

    def execute(self):
        self.device.execute(self.device_execute, layers_span=range(self.start_layer, self.end_layer))
    
    def get_activations(self, top_eigen_vectors, classes_mean, device_information: DeviceInformation, negative_prefix: str, layers_span: List[int]) -> GetActivationsOutput:
        """
        Generate the activations that will then be plotted.
        """
        model = AutoModelForSequenceClassification.from_pretrained(self.ranker_id)
        model.eval()
        if model.config.num_labels == 1:
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

        positive_global_storage = None
        negative_global_storage = None
        weak_negative_global_storage = None

        if negative_prefix == "all":
            principal_components = [0, 1]

        else:
            principal_components = [0]

        for dataset_dict in self.list_of_datasets:
            dataset_name = dataset_dict["dataset_name"]
            logging.info(f"Loading positive pairs at: {self.agregation_output.per_dataset_storage_paths[dataset_name]['positives']}/query-passage_pairs_storage.json")
            with open(f"{self.agregation_output.per_dataset_storage_paths[dataset_name]['positives']}/query-passage_pairs_storage.json", 'r') as f:
                positive_pairs = json.load(f)

            positive_storage = iterate_over_pairs_and_store_activations(
                dataset_name=dataset_name, 
                max_passages_per_query=dataset_dict["max_passages_per_query"],
                pairs=positive_pairs, 
                device=device_information.device,
                tokenizer=tokenizer,
                extractor=extractor,
                target_module=self.target_module,
                layers_span=layers_span,
                principal_components=[0, 1] if negative_prefix == "all" else [0],
                top_eigen_vectors=top_eigen_vectors,
                input_parts=self.input_parts,
                activation_fct=activation_fct,
                pred_position=target_position,
            )

            if positive_global_storage is None:
                positive_global_storage = positive_storage
            else:
                for layer in positive_storage.keys():
                    for key in positive_storage[layer].keys():
                        positive_global_storage[layer][key] = np.concatenate((positive_global_storage[layer][key], positive_storage[layer][key]), axis=0)

            if negative_prefix == "all":
                logging.info("Loading both STRONG and WEAK negatives") 
                logging.info(f"Loading negative pairs at: {self.agregation_output.per_dataset_storage_paths[dataset_name]['strong_negatives']}/query-passage_pairs_storage.json")
                with open(f"{self.agregation_output.per_dataset_storage_paths[dataset_name]['strong_negatives']}/query-passage_pairs_storage.json", 'r') as f:
                    negative_pairs = json.load(f)

                negative_storage = iterate_over_pairs_and_store_activations(
                    dataset_name=dataset_name, 
                    max_passages_per_query=dataset_dict["max_passages_per_query"],
                    pairs=negative_pairs, 
                    device=device_information.device,
                    tokenizer=tokenizer,
                    extractor=extractor,
                    target_module=self.target_module,
                    layers_span=layers_span,
                    principal_components=[0, 1],
                    top_eigen_vectors=top_eigen_vectors,
                    input_parts=self.input_parts,
                    activation_fct=activation_fct,
                    pred_position=target_position,
                )
                if negative_global_storage is None:
                    negative_global_storage = negative_storage
                    
                else:
                    for layer in negative_storage.keys():
                        for key in negative_storage[layer].keys():
                            negative_global_storage[layer][key] = np.concatenate((negative_global_storage[layer][key], negative_storage[layer][key]), axis=0)

                logging.info(f"Loading negative pairs at: {self.agregation_output.per_dataset_storage_paths[dataset_name]['weak_negatives']}/query-passage_pairs_storage.json")
                with open(f"{self.agregation_output.per_dataset_storage_paths[dataset_name]['weak_negatives']}/query-passage_pairs_storage.json", 'r') as f:
                    negative_pairs = json.load(f)

                weak_negative_storage = iterate_over_pairs_and_store_activations(
                    dataset_name=dataset_name, 
                    max_passages_per_query=dataset_dict["max_passages_per_query"],
                    pairs=negative_pairs, 
                    device=device_information.device,
                    tokenizer=tokenizer,
                    extractor=extractor,
                    target_module=self.target_module,
                    layers_span=layers_span,
                    principal_components=[0, 1],
                    top_eigen_vectors=top_eigen_vectors,
                    input_parts=self.input_parts,
                    activation_fct=activation_fct,
                    pred_position=target_position,
                )
                if weak_negative_global_storage is None:
                    weak_negative_global_storage = weak_negative_storage
                else:
                    for layer in weak_negative_storage.keys():
                        for key in weak_negative_storage[layer].keys():
                            weak_negative_global_storage[layer][key] = np.concatenate((weak_negative_global_storage[layer][key], weak_negative_storage[layer][key]), axis=0)
            else:
                logging.info(f"Loading negative pairs at: {self.agregation_output.per_dataset_storage_paths[dataset_name][negative_prefix]}/query-passage_pairs_storage.json")
                with open(f"{self.agregation_output.per_dataset_storage_paths[dataset_name][negative_prefix]}/query-passage_pairs_storage.json", 'r') as f:
                    negative_pairs = json.load(f)

                negative_storage = iterate_over_pairs_and_store_activations(
                    dataset_name=dataset_name, 
                    max_passages_per_query=dataset_dict["max_passages_per_query"],
                    pairs=negative_pairs, 
                    device=device_information.device,
                    tokenizer=tokenizer,
                    extractor=extractor,
                    target_module=self.target_module,
                    layers_span=layers_span,
                    principal_components=[0, 1] if negative_prefix == "all" else [0],
                    top_eigen_vectors=top_eigen_vectors,
                    input_parts=self.input_parts,
                    activation_fct=activation_fct,
                    pred_position=target_position,
                )
                if negative_global_storage is None:
                    negative_global_storage = negative_storage

                else:
                    for layer in negative_storage.keys():
                        for key in negative_storage[layer].keys():
                            negative_global_storage[layer][key] = np.concatenate((negative_global_storage[layer][key], negative_storage[layer][key]), axis=0)

        if negative_prefix == "strong_negatives":
            return GetActivationsOutput(
                positive_activations=positive_global_storage,
                negative_activations=negative_global_storage,
                weak_negative_activations=None,
            )
        elif negative_prefix == "weak_negatives":
            return GetActivationsOutput(
                positive_activations=positive_global_storage,
                negative_activations=None,
                weak_negative_activations=negative_global_storage,
            )
        else:
            return GetActivationsOutput(
                positive_activations=positive_global_storage,
                negative_activations= negative_global_storage,
                weak_negative_activations= weak_negative_global_storage,
            )
        
    def device_execute(self, device_information: DeviceInformation, layers_span: List[int]):        
        with gzip.open(f"{self.collect_direction.storage_path}/strong_negatives/{self.target_module}_top_eigen_vectors.pkl.gz", 'rb') as f:
            top_eigen_vectors = pickle.load(f)

        with gzip.open(f"{self.collect_direction.storage_path}/strong_negatives/{self.target_module}_classes_means.pkl.gz", 'rb') as f:
            classes_mean = pickle.load(f)

        logging.info("Collecting activations for positive and STRONG negatives.")
        get_activations_output = self.get_activations(top_eigen_vectors, classes_mean, device_information, negative_prefix="strong_negatives", layers_span=layers_span)
        logging.info("2) Computing classification metrics for STRONG negatives")
        self.classify_activations(
            get_activations_output.positive_activations, 
            get_activations_output.negative_activations, 
            prefix="strong_negatives"
        )

        if Path(f"{self.collect_direction.storage_path}/weak_negatives/{self.target_module}_top_eigen_vectors.pkl.gz").exists():
            logging.info("Loading eigen vectors for WEAK negatives.")
            with gzip.open(f"{self.collect_direction.storage_path}/weak_negatives/{self.target_module}_top_eigen_vectors.pkl.gz", 'rb') as f:
                weak_top_eigen_vectors = pickle.load(f)

            with gzip.open(f"{self.collect_direction.storage_path}/weak_negatives/{self.target_module}_classes_means.pkl.gz", 'rb') as f:
                weak_classes_mean = pickle.load(f)
            logging.info("Collecting activations for positive and WEAK negatives.")
            get_activations_output = self.get_activations(weak_top_eigen_vectors, weak_classes_mean, device_information, negative_prefix="weak_negatives", layers_span=layers_span)
            logging.info("3) Computing classification metrics for WEAK negatives")
            self.classify_activations(
                get_activations_output.positive_activations, 
                get_activations_output.weak_negative_activations, 
                prefix="weak_negatives"
            )

            logging.info("Loading eigen vectors across ALL data.")
            with gzip.open(f"{self.collect_direction.storage_path}/all/{self.target_module}_top_eigen_vectors.pkl.gz", 'rb') as f:
                all_top_eigen_vectors = pickle.load(f)
            with gzip.open(f"{self.collect_direction.storage_path}/all/{self.target_module}_classes_means.pkl.gz", 'rb') as f:
                all_classes_mean = pickle.load(f)
            logging.info("Collecting activations across ALL relevance levels.")
            get_activations_output = self.get_activations(all_top_eigen_vectors, all_classes_mean, device_information, negative_prefix="all", layers_span=layers_span)
            logging.info("4) Computing classification metrics across ALL relevance levels")
            self.classify_activations_for_all_negatives(
                get_activations_output.positive_activations, 
                get_activations_output.negative_activations, 
                get_activations_output.weak_negative_activations, 
                prefix="all"
            )  
           
    def classify_activations(self, positive_global_storage, negative_global_storage, prefix: str):
        csv_rows = [("Layer", "Input part", "CM",
                     "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_Precision", "Weighted_Recall", "Weighted_F1")]

        for layer in positive_global_storage.keys():
            for input_part in positive_global_storage[layer].keys():
                # Get activations and project them
                pos_proj = positive_global_storage[layer][input_part] 
                neg_proj = negative_global_storage[layer][input_part] 

                X_lda = np.concatenate([pos_proj, neg_proj], axis=0)
                y = np.concatenate([np.full(pos_proj.shape[0], 1), np.full(neg_proj.shape[0], 0)], axis=0)

                clf = LogisticRegression()
                scores = cross_val_score(clf, X_lda, y, cv=5)

                logging.info(f"Layer {layer}, Input part {input_part} - Cross-validated accuracy: {scores.mean()}")

                X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                cm = confusion_matrix(y_test, y_pred)
                cm_flat = cm.flatten()
                
                # Classification Report (averages only)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Store row
                csv_rows.append((layer, input_part, cm_flat, 
                                 report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'],
                                 report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']
                ))


        # Save results to CSV
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)

        csv_path = os.path.join(self.storage_path, f"{prefix}_{self.target_module}_classification_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        logging.info(f"Classification results saved to {csv_path}")

    def classify_activations_for_all_negatives(self, positive_global_storage, negative_global_storage, weak_negative_global_storage, prefix: str):
        csv_rows = [("Layer", "Input part", "CM",
                     "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_Precision", "Weighted_Recall", "Weighted_F1"
        )]

        for layer in positive_global_storage.keys():
            for input_part in positive_global_storage[layer].keys():
                # Get activations and project them
                pos_proj = positive_global_storage[layer][input_part] 
                neg_proj = negative_global_storage[layer][input_part] 
                weak_neg_proj = weak_negative_global_storage[layer][input_part] 
                X_lda = np.concatenate([pos_proj, neg_proj, weak_neg_proj], axis=0)
                y = np.concatenate([np.full(pos_proj.shape[0], 1), np.full(neg_proj.shape[0], 0), np.full(weak_neg_proj.shape[0], -1)], axis=0)

                clf = LogisticRegression()
                scores = cross_val_score(clf, X_lda, y, cv=5)

                logging.info(f"Layer {layer}, Input part {input_part} - Cross-validated accuracy: {scores.mean()}")

                X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                cm = confusion_matrix(y_test, y_pred)
                cm_flat = cm.flatten()
                
                # Classification Report (averages only)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Store row
                csv_rows.append((layer, input_part, cm_flat, 
                                 report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'],
                                 report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']
                ))

        # Save results to CSV
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)

        csv_path = os.path.join(self.storage_path, f"{prefix}_{self.target_module}_classification_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        logging.info(f"Classification results saved to {csv_path}")

class AggregateClassificationResults(Task):
    list_of_classifications: Param[List[ComputeClassification]]

    output_path: Annotated[Path, pathgenerator("classification_results")]

    def execute(self):
        aggregated = {
            'weak_negatives': [],
            'strong_negatives': [],
            'all': []
        }

        # Pattern to extract table type and model part
        pattern = re.compile(r'^(weak_negatives|strong_negatives|all)_(.+)_classification_results\.csv$')

        for task in self.list_of_classifications:
            if not os.path.isdir(task.storage_path):
                print(f"Skipping {task.storage_path}: Not a valid directory.")
                continue

            for filename in os.listdir(task.storage_path):
                if not filename.endswith('.csv'):
                    continue

                match = pattern.match(filename)
                if not match:
                    print(f"Skipping {filename}: Doesn't match expected pattern.")
                    continue

                table_type, model_part = match.groups()
                file_path = os.path.join(task.storage_path, filename)

                try:
                    df = pd.read_csv(file_path)
                    df['ModelPart'] = model_part
                    aggregated[table_type].append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        os.makedirs(self.output_path, exist_ok=True)

        # Save each aggregated DataFrame
        for key, dfs in aggregated.items():
            if dfs:
                output_path = os.path.join(self.output_path, f'aggregated_{key}.csv')
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(output_path, index=False)
                print(f"Saved {key} data to {output_path}")
            else:
                print(f"No data found for {key}, skipping export.")
