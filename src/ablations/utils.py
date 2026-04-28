from collections import defaultdict
import os
from ir_measures import nDCG
import ir_measures
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

    version: Constant[int] = 7

    def execute(self):
        logging.info("Get all qrels.")
        dict_qrels = get_relevance_levels(self.dataset_name, self.parsed_dataset)

        logging.info("Computing the worst possible ranking.")
        worst_results_per_query = {}
        for query_id, doc_rels in dict_qrels.items():
            worst_results_per_query[query_id] = {doc_id: 1.0 / (rel + 1) + np.random.uniform(0, 1e-9) for doc_id, rel in doc_rels.items()}
        # worst_ndcg_score_per_query = {"ndcg": {m.query_id: m.value for m in ir_measures.iter_calc([nDCG@10], dict_qrels, worst_results_per_query)}}
        worst_metric = ir_measures.calc_aggregate([nDCG@10], dict_qrels, worst_results_per_query)

        logging.info("Computing random rankings.")
        N_RUNS = 1000
        random_ndcg_per_query = defaultdict(list)

        for _ in range(N_RUNS):
            random_results_per_query = {
                query_id: {doc_id: np.random.uniform(0, 1) for doc_id in doc_rels}
                for query_id, doc_rels in dict_qrels.items()
            }
            for m in ir_measures.iter_calc([nDCG@10], dict_qrels, random_results_per_query):
                random_ndcg_per_query[m.query_id].append(m.value)

        # Average per-query scores across runs
        random_ndcg_score_per_query = {"ndcg": {query_id: np.mean(scores) for query_id, scores in random_ndcg_per_query.items()}}
        random_metric = {nDCG@10: np.mean(list(random_ndcg_score_per_query["ndcg"].values()))}

        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        pd.DataFrame.from_dict(data=random_metric, orient='index').to_csv(f"{self.storage_path}/random_results.csv", header=True)
        pd.DataFrame.from_dict(data=worst_metric, orient='index').to_csv(f"{self.storage_path}/worst_results.csv", header=True)

        logging.info(f"Dataset: {self.dataset_name} - Worst ranking nDCG@10: {worst_metric[nDCG@10]:.2f} - Random ranking nDCG@10: {random_metric[nDCG@10]:.2f}")