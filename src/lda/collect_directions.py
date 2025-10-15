from experimaestro import Task, Param, Constant
from scipy.linalg import eigh
from typing import Dict, List, Union, Annotated
from dataclasses import dataclass
from pathlib import Path
from src.lda.generate_data import AgregateMatricesOutput
from experimaestro.generators import pathgenerator
import numpy as np
import os
import pickle
import gzip


import logging
logging.basicConfig(level=logging.INFO)
@dataclass
class GetActivationsOutput:
    positive_activations: Dict
    negative_activations: Union[Dict, None]
    weak_negative_activations: Union[Dict, None]

FILTER_WORDS = ["what", "who", "when", "where", "why", "which", "how", "do", "example"]

MARKERS = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', '*']
POSITIVE_COLOR = '#e66101'
STRONG_NEGATIVE_COLOR = '#5e3c99'
WEAK_NEGATIVE_COLOR = '#1a9641'

class CollectDirections(Task):
    agregation_output: Param[AgregateMatricesOutput]

    target_modules: Param[List[str]]

    input_parts: Param[List[str]]

    start_layer: Param[int]

    end_layer: Param[int]

    version: Constant[int] = 6

    storage_path: Annotated[Path, pathgenerator("pca_outputs")]

    def execute(self):
        """ Similar to the PCA analysis, except we also compute the inner class statistics and not only across the whole dataset
        to obtain the different components of the LDA.
        """
        with gzip.open(f"{self.agregation_output.task.positive_storage_path}/global_storage.pkl.gz", 'rb') as f:
            positive_data = pickle.load(f)
        with gzip.open(f"{self.agregation_output.task.positive_storage_path}/denominator_sum.pkl.gz", 'rb') as f:
            positive_denominator_dict = pickle.load(f)

        with gzip.open(f"{self.agregation_output.task.negative_storage_path}/global_storage.pkl.gz", 'rb') as f:
            negative_data = pickle.load(f)
        with gzip.open(f"{self.agregation_output.task.negative_storage_path}/denominator_sum.pkl.gz", 'rb') as f:
            negative_denominator_dict = pickle.load(f)

        logging.info("### \n Computing LDA for Positive AND STRONG Negative combined. \n ###")
        self.get_lda_for_two_classes(positive_data, negative_data, positive_denominator_dict, negative_denominator_dict, is_weak=False, positive_only=False)

        weak_negatives_path = Path(f"{self.agregation_output.task.negative_storage_path.parent}/weak_negative")
        if weak_negatives_path.exists():
            logging.info("Get the data from weak negatives.")
            with gzip.open(f"{self.agregation_output.task.negative_storage_path.parent}/weak_negative/global_storage.pkl.gz", 'rb') as f:
                weak_negative_data = pickle.load(f)
            with gzip.open(f"{self.agregation_output.task.negative_storage_path.parent}/weak_negative/denominator_sum.pkl.gz", 'rb') as f:
                weak_negative_denominator_dict = pickle.load(f)

            logging.info("### \n Computing LDA for Positive AND WEAK Negative combined. \n ###")
            self.get_lda_for_two_classes(positive_data, weak_negative_data, positive_denominator_dict, weak_negative_denominator_dict, is_weak=True, positive_only=False)

            # Compute PCA across all datasets
            logging.info("### \n Computing LDA for Positive AND ALL Negative combined. \n ###")
            self.get_lda_for_three_classes(
                positive_data, 
                negative_data,
                weak_negative_data, 
                positive_denominator_dict, 
                negative_denominator_dict,
                weak_negative_denominator_dict
            )

    def regularize_matrix(self, matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """ Regularizes the matrix to make it positive definite by adding a small value to the diagonal.
        """
        i = 0
        eigvals = np.linalg.eigvalsh(matrix)

        new_epsilon = epsilon
        new_matrix = matrix
        while not np.all(eigvals > 0):
            # Check that matrix is positive definite, if not we regularize it
            i += 1
            new_epsilon = i * epsilon
            new_matrix = matrix + new_epsilon * np.eye(matrix.shape[0])
            eigvals = np.linalg.eigvalsh(new_matrix)
        if i > 0:
            logging.info(f"Regularized the matrix with epsilon {new_epsilon} to make it positive definite.")
        else:
            logging.info("Matrix was already positive definite.")
        return new_matrix

    def get_lda_for_two_classes(self, positive_data, negative_data, positive_denominator_dict, negative_denominator_dict, is_weak: bool = False, positive_only: bool = False):
        # First, merge the parts corresponding to the different tokens of the query for each layer and each module
        initial_dict = {part: 0 for part in self.input_parts}
        for target_module in self.target_modules:   
            separability_dict = dict()      
            classes_means = dict()
            top_eigen_vector_storage = dict()
            top_eigen_value_storage = dict()
            logging.info(f"*** Target module: {target_module} ***")
            for layer_nb in range(self.start_layer, self.end_layer):
                separability_dict[f"layer_{layer_nb}"] = dict()
                classes_means[f"layer_{layer_nb}"] = dict()
                top_eigen_vector_storage[f"layer_{layer_nb}"] = dict()
                top_eigen_value_storage[f"layer_{layer_nb}"] = dict()
                # Aggregattion for the between class scatter
                combined_numerator = initial_dict.copy()
                combined_weighted_sum = initial_dict.copy()
                combined_denominator = initial_dict.copy()
                # Aggregations for the within class scatter

                positive_numerator = initial_dict.copy()
                positive_weighted_sum = initial_dict.copy()
                positive_denominator = initial_dict.copy()

                negative_numerator = initial_dict.copy()
                negative_weighted_sum = initial_dict.copy()
                negative_denominator = initial_dict.copy()

                # Now we keep the query tokens separated
                for input_part in self.input_parts:
                    combined_numerator[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"].astype(np.float64) + negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"].astype(np.float64)
                    combined_weighted_sum[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"].astype(np.float64) + negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"].astype(np.float64)
                    combined_denominator[input_part] += 2 # positive_denominator_dict[input_part].astype(np.float64) + negative_denominator_dict[input_part].astype(np.float64)

                    positive_numerator[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"].astype(np.float64)
                    positive_weighted_sum[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"].astype(np.float64)
                    positive_denominator[input_part] += 1 # positive_denominator_dict[input_part].astype(np.float64)

                    negative_numerator[input_part] += negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"].astype(np.float64)
                    negative_weighted_sum[input_part] += negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"].astype(np.float64)
                    negative_denominator[input_part] += 1 # negative_denominator_dict[input_part].astype(np.float64)

                # Then we iterate again to compute the eigenvectors and eigeinvalues
                for key in combined_numerator.keys():
                    if combined_denominator[key] != 0:
                        positive_estimated_variance = positive_numerator[key] / positive_denominator[key]
                        positive_mean = positive_weighted_sum[key] / positive_denominator[key]
                        positive_class_scatter = positive_estimated_variance - np.outer(positive_mean, positive_mean)

                        negative_estimated_variance = negative_numerator[key] / negative_denominator[key]
                        negative_mean = negative_weighted_sum[key] / negative_denominator[key]
                        negative_class_scatter = negative_estimated_variance -  np.outer(negative_mean, negative_mean)

                        combined_mean = combined_weighted_sum[key] / combined_denominator[key]
                        between_class_scatter = positive_denominator[key] * np.outer(positive_mean - combined_mean, positive_mean - combined_mean) + negative_denominator[key] * np.outer(negative_mean - combined_mean, negative_mean - combined_mean)

                        within_class_scatter = positive_class_scatter + negative_class_scatter
                        regularized_within_class = self.regularize_matrix(within_class_scatter)
                        try:
                            S, eigenvectors = eigh(between_class_scatter, regularized_within_class)
                        except ValueError as e:
                            logging.error(f"Eigenvalue computation failed for layer {layer_nb}, module {target_module}, key {key}. Skipping.")
                            separability_dict[f"layer_{layer_nb}"][key] = [None, None]
                            classes_means[f"layer_{layer_nb}"][key] = {"positive_mean": None, "negative_mean": None}
                            continue
                        sorted_S = -np.sort(-S)
                        sorted_eigenvectors = eigenvectors[:, np.argsort(-S)]
                        top_eigen_vector_storage[f"layer_{layer_nb}"][key] = sorted_eigenvectors[:,:]
                        top_eigen_value_storage[f"layer_{layer_nb}"][key] = sorted_S[:]
                        logging.info(f"Layer {layer_nb}, **{key}** / Separability estimation by the first axis: {sum(sorted_S[:1]) / sum(sorted_S[:])} / 2 axes: {sum(sorted_S[:2]) / sum(sorted_S[:])}")    
                        separability_dict[f"layer_{layer_nb}"][key] = [sum(sorted_S[:1]) / sum(sorted_S[:]), sum(sorted_S[:2]) / sum(sorted_S[:])]
                        classes_means[f"layer_{layer_nb}"][key] = {"positive_mean": positive_mean, "negative_mean": negative_mean}
                        
            logging.info(f"Saving the results to pickle files.")
            if not Path(self.storage_path).exists():
                os.mkdir(self.storage_path)

            if is_weak:
                path = Path(f"{self.storage_path}/weak_negatives")
            elif positive_only:
                path = Path(f"{self.storage_path}/positives_only")
            else:
                path = Path(f"{self.storage_path}/strong_negatives")

            if not path.exists():
                os.mkdir(path)
            with gzip.open(f"{path}/{target_module}_top_eigen_vectors.pkl.gz", 'wb') as f:
                pickle.dump(top_eigen_vector_storage, f)
            with gzip.open(f"{path}/{target_module}_top_eigen_values.pkl.gz", 'wb') as f:
                pickle.dump(top_eigen_value_storage, f)
            with gzip.open(f"{path}/{target_module}_separability_logs.pkl.gz", 'wb') as f:
                pickle.dump(separability_dict, f)
            with gzip.open(f"{path}/{target_module}_classes_means.pkl.gz", 'wb') as f:
                pickle.dump(classes_means, f)

    def get_lda_for_three_classes(
        self, 
        positive_data, 
        negative_data,
        weak_negative_data, 
        positive_denominator_dict, 
        negative_denominator_dict, 
        weak_negative_denominator_dict,
    ):
        # First, merge the parts corresponding to the different tokens of the query for each layer and each module
        initial_dict = {part: 0 for part in self.input_parts}
        for target_module in self.target_modules: 
            separability_dict = dict()        
            top_eigen_vector_storage = dict()
            top_eigen_value_storage = dict()
            classes_means = dict()
            logging.info(f"*** Target module: {target_module} ***")
            for layer_nb in range(self.start_layer, self.end_layer):
                top_eigen_vector_storage[f"layer_{layer_nb}"] = dict()
                top_eigen_value_storage[f"layer_{layer_nb}"] = dict()
                classes_means[f"layer_{layer_nb}"] = dict()
                separability_dict[f"layer_{layer_nb}"] = dict()
                combined_numerator = initial_dict.copy()
                combined_weighted_sum = initial_dict.copy()
                combined_denominator = initial_dict.copy()

                # Aggregations for the within class scatter
                positive_numerator = initial_dict.copy()
                positive_weighted_sum = initial_dict.copy()
                positive_denominator = initial_dict.copy()

                negative_numerator = initial_dict.copy()
                negative_weighted_sum = initial_dict.copy()
                negative_denominator = initial_dict.copy()

                weak_negative_numerator = initial_dict.copy()
                weak_negative_weighted_sum = initial_dict.copy()
                weak_negative_denominator = initial_dict.copy()
                # Keep query tokens separated
                for input_part in self.input_parts:
                    combined_numerator[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"] + negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"] + weak_negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"]
                    combined_weighted_sum[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"] + negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"] + weak_negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"]
                    combined_denominator[input_part] += 3 # positive_denominator_dict[input_part] + negative_denominator_dict[input_part] + weak_negative_denominator_dict[input_part]

                    positive_numerator[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"]
                    positive_weighted_sum[input_part] += positive_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"]
                    positive_denominator[input_part] += 1 # positive_denominator_dict[input_part]

                    negative_numerator[input_part] += negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"]
                    negative_weighted_sum[input_part] += negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"]
                    negative_denominator[input_part] += 1 # negative_denominator_dict[input_part]

                    weak_negative_numerator[input_part] += weak_negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_numerator"]
                    weak_negative_weighted_sum[input_part] += weak_negative_data[f"layer_{layer_nb}"][f"{target_module}"][input_part]["streamed_mean"]
                    weak_negative_denominator[input_part] += 1 # weak_negative_denominator_dict[input_part]

                # Then we iterate again to compute the eigenvectors and eigeinvalues
                for key in combined_numerator.keys():
                    if combined_denominator[key] != 0:
                        positive_estimated_variance = positive_numerator[key] / positive_denominator[key]
                        positive_mean = positive_weighted_sum[key] / positive_denominator[key]
                        positive_class_scatter = positive_estimated_variance - np.outer(positive_mean, positive_mean)

                        negative_estimated_variance = negative_numerator[key] / negative_denominator[key]
                        negative_mean = negative_weighted_sum[key] / negative_denominator[key]
                        negative_class_scatter = negative_estimated_variance -  np.outer(negative_mean, negative_mean)

                        weak_negative_estimated_variance = weak_negative_numerator[key] / weak_negative_denominator[key]
                        weak_negative_mean = weak_negative_weighted_sum[key] / weak_negative_denominator[key]
                        weak_negative_class_scatter = weak_negative_estimated_variance - np.outer(weak_negative_mean, weak_negative_mean)

                        combined_mean = combined_weighted_sum[key] / combined_denominator[key]
                        between_class_scatter = positive_denominator[key] * np.outer(positive_mean - combined_mean, positive_mean - combined_mean) + negative_denominator[key] * np.outer(negative_mean - combined_mean, negative_mean - combined_mean) + weak_negative_denominator[key] * np.outer(weak_negative_mean - combined_mean, weak_negative_mean - combined_mean)
                        
                        within_class_scatter = positive_class_scatter + negative_class_scatter + weak_negative_class_scatter
                        
                        regularized_within_class = self.regularize_matrix(within_class_scatter)
                        try:
                            S, eigenvectors = eigh(between_class_scatter, regularized_within_class)
                        except ValueError as e:
                            logging.error(f"Eigenvalue computation failed for layer {layer_nb}, module {target_module}, key {key}. Skipping.")
                            separability_dict[f"layer_{layer_nb}"][key] = [None, None]
                            classes_means[f"layer_{layer_nb}"][key] = {"positive_mean": None, "negative_mean": None, "weak_negative_mean": None}
                            continue
                        sorted_S = -np.sort(-S)
                        sorted_eigenvectors = eigenvectors[:, np.argsort(-S)]
                        top_eigen_vector_storage[f"layer_{layer_nb}"][key] = sorted_eigenvectors[:,:]
                        top_eigen_value_storage[f"layer_{layer_nb}"][key] = sorted_S[:]
                        logging.info(f"Layer {layer_nb}, **{key}** / Separability estimation by the first axis: {sum(sorted_S[:1]) / sum(sorted_S[:])} / First 2 axes: {sum(sorted_S[:2]) / sum(sorted_S[:])}")    
                        separability_dict[f"layer_{layer_nb}"][key] = [sum(sorted_S[:1]) / sum(sorted_S[:]), sum(sorted_S[:2]) / sum(sorted_S[:])]
                        classes_means[f"layer_{layer_nb}"][key] = {"positive_mean": positive_mean, "negative_mean": negative_mean, "weak_negative_mean": weak_negative_mean}

            logging.info(f"Saving the results to pickle files.")
            if not Path(self.storage_path).exists():
                os.mkdir(self.storage_path)

            if not Path(f"{self.storage_path}/all").exists():
                os.mkdir(f"{self.storage_path}/all")
            with gzip.open(f"{self.storage_path}/all/{target_module}_top_eigen_vectors.pkl.gz", 'wb') as f:
                pickle.dump(top_eigen_vector_storage, f)
            with gzip.open(f"{self.storage_path}/all/{target_module}_top_eigen_values.pkl.gz", 'wb') as f:
                pickle.dump(top_eigen_value_storage, f)
            with gzip.open(f"{self.storage_path}/all/{target_module}_separability_logs.pkl.gz", 'wb') as f:
                pickle.dump(separability_dict, f)
            with gzip.open(f"{self.storage_path}/all/{target_module}_classes_means.pkl.gz", 'wb') as f:
                pickle.dump(classes_means, f)
