import os
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import torch
from typing import Annotated, Callable, Optional

from utils import get_relevance_levels, untuple
from experimaestro import Task, Param, Config, Meta, Annotated, pathgenerator, Constant
from experimaestro.generators import pathgenerator
from pathlib import Path

from datamaestro import prepare_dataset
from datamaestro_text.data.ir import PairwiseSampleDataset

import logging

logging.basicConfig(level=logging.INFO)    

class AblationOutput(Config):
    __xpmid__="src.ablations.ablation_utils.ablationoutput"
    
    task: Meta[Config]

class AgregationOutput(Config):
    __xpmid__="src.ablations.ablation_utils.agregationoutput"
    
    task: Meta[Config]

class PrunedModelForCrossScorer(torch.nn.Module):
    # Hook used to mask the output of some layer given a pruning scheme
    def __init__(self, model: torch.nn.Module, start_layer: int, end_layer: int):
        super().__init__()
        self.model = model
        self.hooks_handles = list()
        self.attention_scores_mask = None
        self.start_layer = start_layer
        self.end_layer = end_layer

        for layer_nb in range(self.start_layer, self.end_layer):
            layer = dict([*self.model.named_modules()])[f"bert.encoder.layer.{layer_nb}.attention.self.dropout"]
            self.hooks_handles.append(layer.register_forward_hook(self.prune_dropout()))

    def prune_dropout(self) -> Callable:
        def hook(model, input, output):
            assert self.attention_scores_mask is not None, "Attention scores mask must be set before calling forward"
            output = untuple(output)
            masked_output = output * self.attention_scores_mask
            return masked_output
        return hook
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()

    def forward(
        self, 
        attention_scores_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        self.attention_scores_mask = attention_scores_mask
        model_outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
        self.attention_scores_mask = None # reset it to None to make sure we don't use twice the same mask
        return model_outputs        

class PerspectiveAblationStudy(Task):
    __xpmid__="src.ablations.ablation_utils.perspectiveablationstudy"

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]

    storage_path: Annotated[Path, pathgenerator("storage")]

    version: Constant[int] = 2

    def execute(self):
        shift = 1 # Make the shift parametrizable in case we want to consider the easy negatives

        logging.info("Get all qrels.")
        assessments = prepare_dataset(self.dataset_name).assessments
        base_qrels = {
            assessedTopic.topic_id: {r.doc_id: r.rel for r in assessedTopic.assessments}
            for assessedTopic in assessments.iter()
        }
        target_qrels = {}
        logging.info("Computing the worst possible ranking.")
        worst_ranking = dict()
        for query_id in base_qrels.keys():
            worst_ranking[query_id] = dict()
            for doc_id, relevance in base_qrels[query_id].items():
                worst_ranking[query_id][doc_id] = 1 / (relevance + shift)
            target_qrels[query_id] = base_qrels[query_id]

        queries = list(base_qrels.keys())
        documents = list({doc for query in target_qrels for doc in target_qrels[query]})
        
        qrels_scores = np.array([
                    [target_qrels[query].get(doc, 0) for doc in documents]
                    for query in queries
                ])
        # Gather worst rankings
        worst_ranking_per_query = np.array([
            [worst_ranking[query].get(doc, 0) for doc in documents]
            for query in queries
        ])
        # Compute NDCG scores for each query
        worst_metric = {"nDCG@10": ndcg_score(qrels_scores, worst_ranking_per_query, k=10, ignore_ties=False)}

        logging.info("Computing random rankings.")
        random_ranking = dict()
        for query_id in target_qrels.keys():
            random_ranking[query_id] = dict()
            for doc_id, relevance in target_qrels[query_id].items():
                random_ranking[query_id][doc_id] = 1.0

        # Gather random rankings
        random_ranking_per_query = np.array([
            [random_ranking[query].get(doc, 0) for doc in documents]
            for query in queries
        ])
        # Compute NDCG scores for each query
        random_metric = {"nDCG@10": ndcg_score(qrels_scores, random_ranking_per_query, k=10, ignore_ties=False)}

        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        pd.DataFrame.from_dict(data=random_metric, orient='index').to_csv(f"{self.storage_path}/random_results.csv", header=True)
        pd.DataFrame.from_dict(data=worst_metric, orient='index').to_csv(f"{self.storage_path}/worst_results.csv", header=True)