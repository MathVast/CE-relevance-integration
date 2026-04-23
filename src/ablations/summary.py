from .utils import AblationOutput, AgregationOutput
from datamaestro import prepare_dataset
import csv
import os
import json
from typing import Callable, Dict, List, Annotated
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from experimaestro import Task, Param, Config, Constant
from experimaestro.generators import pathgenerator
from pathlib import Path
import ir_measures
from ir_measures import *
from scipy import stats

import logging

logging.basicConfig(level=logging.INFO)


class AgregationAblations(Task):
    """This task is designed to agregate the ablation results for a SINGLE dataset and a SINGLE ablation pattern.
    It will write into different files the results of the ablation of some directions of the attention at each layer of a model,
    averaged over the queries.

    :param ablations_output: The output of the ablations tasks over the dataset for this attention pattern.
    """
    __xpmid__="src.ablations.ablation_study.agregationablations"

    ablations_output: Param[List[AblationOutput]]

    storage_path: Annotated[Path, pathgenerator("storage")]

    dataset_name: Param[str]

    target_label: Param[int]

    version: Constant[int] = 2

    def task_outputs(self, dep: Callable[[Config], None]) -> AgregationOutput:
        return dep(AgregationOutput.C(task=self))

    def execute(self):
        assessments = prepare_dataset(self.dataset_name).assessments
        base_qrels = {
            assessedTopic.topic_id: {r.doc_id: r.rel for r in assessedTopic.assessments}
            for assessedTopic in assessments.iter()
        }
        target_qrels = {}
        agregation_results = dict()
        original_results_per_query = dict()
        ablated_results_per_direction_per_query = dict()
        
        for ablation_output in self.ablations_output:
            hash_id = str(ablation_output.task.output_path).split("/")[-2]
            ablation_results = np.load(f"{ablation_output.task.output_path}/averaged_diff.npy")
            if hasattr(ablation_output.task, "cut_attention_from"):
                if isinstance(ablation_output.task.cut_attention_from, list):
                    string_attention_from = "+".join([direction for direction in ablation_output.task.cut_attention_from]) 
                    if hasattr(ablation_output.task, "cut_attention_to"):
                        string_attention_to = "+".join([direction for direction in ablation_output.task.cut_attention_to])
                        key = f"{hash_id}_{string_attention_from}-->{string_attention_to}_L{ablation_output.task.start_layer}_L{ablation_output.task.end_layer}"
                    else:
                        key = f"{hash_id}_{string_attention_from}_L{ablation_output.task.start_layer}_L{ablation_output.task.end_layer}"
            else:
                raise ValueError("Ablation on nothing.")

            agregation_results[key] = ablation_results
            if Path(f"{ablation_output.task.output_path}/ablation_logs.jsonl").exists():
                logging.info(f"Processing logs for {key} from {ablation_output.task.output_path}/ablation_logs.jsonl")
                with open(f"{ablation_output.task.output_path}/ablation_logs.jsonl", "r") as f:
                    for line in f:
                        dico = json.loads(line)
                        if key not in ablated_results_per_direction_per_query.keys():
                            ablated_results_per_direction_per_query[key] = dict()

                        if dico["query_id"] not in ablated_results_per_direction_per_query[key].keys():
                            ablated_results_per_direction_per_query[key][dico["query_id"]] = dict()
                        ablated_results_per_direction_per_query[key][dico["query_id"]][dico["passage_id"]] = float(dico["ablation_logits"][0][self.target_label])
                        if dico["query_id"] not in original_results_per_query.keys():
                            original_results_per_query[dico["query_id"]] = dict()
                        original_results_per_query[dico["query_id"]][dico["passage_id"]] = float(dico["original_logits"][0][self.target_label])
                        target_qrels[dico["query_id"]] = base_qrels[dico["query_id"]]

            elif Path(f"{ablation_output.task.output_path}/ablation_logs.npy").exists(): 
                logging.info(f"Processing logs for {key} from {ablation_output.task.output_path}/ablation_logs.npy")
                activations = np.load(f"{ablation_output.task.output_path}/ablation_logs.npy", allow_pickle=True)
                for dico in activations:
                    if key not in ablated_results_per_direction_per_query.keys():
                        ablated_results_per_direction_per_query[key] = dict()

                    if dico["query_id"] not in ablated_results_per_direction_per_query[key].keys():
                        ablated_results_per_direction_per_query[key][dico["query_id"]] = dict()
                    ablated_results_per_direction_per_query[key][dico["query_id"]][dico["passage_id"]] = float(dico["ablation_logits"][0][self.target_label])
                    if dico["query_id"] not in original_results_per_query.keys():
                        original_results_per_query[dico["query_id"]] = dict()
                    original_results_per_query[dico["query_id"]][dico["passage_id"]] = float(dico["original_logits"][0][self.target_label])
                    target_qrels[dico["query_id"]] = base_qrels[dico["query_id"]]
            else:
                raise ValueError(f"No logs found for {key} in {ablation_output.task.output_path}.")

        base_run_metrics_per_qid = {"ndcg": {m.query_id: m.value for m in ir_measures.iter_calc([nDCG@10], target_qrels, original_results_per_query)}}
        base_run_metrics = ir_measures.calc_aggregate([nDCG@10], target_qrels, original_results_per_query)

        ablated_run_metrics_per_direction_per_qid = dict()
        ablated_run_metrics_per_direction= dict()
        ndcg_p_value_per_direction = dict()
        for direction in ablated_results_per_direction_per_query.keys():
            queries = list(target_qrels.keys())
            documents = list({doc for query in target_qrels for doc in target_qrels[query]})

            qrels_scores = np.array([
                [target_qrels[query].get(doc, 0) for doc in documents]
                for query in queries
            ])
            runs = np.array([
                [ablated_results_per_direction_per_query[direction][query].get(doc, 0) for doc in documents]
                for query in queries
            ])
            # Compute NDCG scores for each query
            ndcg_scores = []
            for qrel_score, single_run in zip(qrels_scores, runs):
                ndcg_scores.append(ndcg_score([qrel_score], [single_run], k=10, ignore_ties=False))

            # Compute NDCG scores for each query
            ablated_run_metrics_per_direction_per_qid[direction] = {"ndcg": {query: score for query, score in zip(queries, ndcg_scores)}}
            ablated_run_metrics_per_direction[direction] = ndcg_score(qrels_scores, runs, k=10, ignore_ties=False)

            ndcg_t_statistic, ndcg_p_value = stats.ttest_ind(
                [base_run_metrics_per_qid["ndcg"][v] for v in target_qrels.keys()], 
                [ablated_run_metrics_per_direction_per_qid[direction]["ndcg"][v] for v in target_qrels.keys()]
            )
            ndcg_p_value_per_direction[direction] = ndcg_p_value
                
        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        pd.DataFrame.from_dict(data=agregation_results, orient='index').to_csv(f"{self.storage_path}/agregation_results.csv", header=True)
        pd.DataFrame.from_dict(data=base_run_metrics, orient='index').to_csv(f"{self.storage_path}/base_run_metrics.csv", header=True)
        pd.DataFrame.from_dict(data=ablated_run_metrics_per_direction, orient='index').to_csv(f"{self.storage_path}/ablated_run_metrics.csv", header=True)
        pd.DataFrame.from_dict(data=ndcg_p_value_per_direction, orient='index').to_csv(f"{self.storage_path}/ndcg_p_values.csv", header=True)

class SummaryAblationStudies(Task):
    """On the conrtary, this task aims at summarizing ablations results ACROSS datasets. It returns the average 
    results for each layer and each attention pattern over multiple datasets.

    :param ablations_output: The output of the ablations tasks per dataset for this attention pattern.
    """
    __xpmid__="src.ablations.ablation_study.summaryablationstudies"

    agregation_outputs: Param[Dict[str, AgregationOutput]]

    storage_path: Annotated[Path, pathgenerator("storage")]

    def execute(self):
        agregation_results = dict()
        count = 0
        for _, agregation_output in self.agregation_outputs.items():
            df = pd.read_csv(f"{agregation_output.task.storage_path}/base_run_metrics.csv", index_col=0)
            ndcg_baseline = df.iloc[0,0] # Get the nDCG@10 scores of the model, without any ablation
            with open(f"{agregation_output.task.storage_path}/ablated_run_metrics.csv", 'r', newline='') as csv_file:
                ablated_results = dict()
                reader = csv.reader(csv_file)
                next(reader)  # Skip the header row (the one with just ",0")

                for row in reader:
                    if len(row) == 2:
                        key = row[0].strip()
                        parsed_key = key.split("_", 1)[1]
                        value = float(row[1].strip())
                        ablated_results[parsed_key] = value
                        
            relative_degradations = {
                k: (ndcg_baseline - v) / ndcg_baseline
                for k, v in ablated_results.items()
            }
            
            if count == 0:
                for key in relative_degradations.keys():
                    agregation_results[key] = [relative_degradations[key]]
            else:
                for key in relative_degradations.keys():
                    agregation_results[key].append(relative_degradations[key])
            count += 1
            
        # Compute the average and standard deviation for each key
        for key in agregation_results.keys():
            values = agregation_results[key]
            mean = np.mean(values)
            std = np.std(values)
            agregation_results[key] = {
                "mean": mean,
                "std": std
            }
        # Save the results to a CSV file
        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        pd.DataFrame.from_dict(data=agregation_results, orient='index').to_csv(f"{self.storage_path}/summary_results.csv", header=True)
        logging.info(f"Summary results saved to {self.storage_path}/summary_results.csv")